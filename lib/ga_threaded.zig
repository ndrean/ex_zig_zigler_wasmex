const beam = @import("beam");
const std = @import("std");
const math = std.math;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Random = std.Random;
const Thread = std.Thread;
const native_endian = @import("builtin").target.cpu.arch.endian();

const POPULATION_SIZE: comptime_int = 100;
const GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!\"#%&/()=?@${[]}";
const MAX_TARGET_LENGTH: comptime_int = 50;
const MAX_GENERATIONS: comptime_int = 5_000; // Prevent infinite loops
const TOURNAMENT_SIZE = 10;
const GENE_MIX: comptime_float = 0.45;
const NB_TRIALS_PER_THREAD: comptime_int = 4;

var beam_allocator = beam.allocator;

const ThreadContext = struct {
    allocator: std.mem.Allocator,
    elitism: usize,
    results_ptr: []usize,
    input: []const u8,
};

// New ThreadPool structure for managing parallel trials
const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []Thread,
    results: []usize,
    elitism: usize,

    fn init(args: struct {
        allocator: std.mem.Allocator,
        num_threads: usize,
        elitism: usize,
    }) !ThreadPool {
        const results = try args.allocator.alloc(usize, args.num_threads * NB_TRIALS_PER_THREAD);
        const threads = try args.allocator.alloc(Thread, args.num_threads);

        return ThreadPool{
            .allocator = args.allocator,
            .threads = threads,
            .results = results,
            .elitism = args.elitism,
        };
    }

    fn deinit(self: *ThreadPool) void {
        self.allocator.free(self.threads);
        self.allocator.free(self.results);
    }

    fn runParallelTrials(self: *ThreadPool, input: []const u8) !void {
        for (0..self.threads.len) |i| {
            // we build a slice of the results array for each thread
            const thread_results = self.results[i * NB_TRIALS_PER_THREAD .. (i + 1) * NB_TRIALS_PER_THREAD];

            const thread_context = ThreadContext{
                .allocator = self.allocator,
                .elitism = self.elitism,
                .results_ptr = thread_results, // a slice
                .input = input,
            };

            self.threads[i] = try Thread.spawn(
                .{},
                threadTrialRunner,
                .{thread_context},
            );
        }

        // Join all threads
        for (self.threads) |*thread| {
            thread.join();
        }
    }
};

fn threadTrialRunner(context: ThreadContext) !void {
    var rand = Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
    const ran = rand.random();

    for (0..NB_TRIALS_PER_THREAD) |j| {
        const trial_count = try runGeneticAlgorithm(
            context.allocator,
            context.elitism,
            ran,
            context.input,
        );
        // updates the slice
        context.results_ptr[j] = @min(MAX_GENERATIONS, trial_count);
    }
}

const Individual = struct {
    string: []u8,
    fitness: usize,

    fn init(allocator: std.mem.Allocator, string: []const u8, fitness: ?usize) !Individual {
        const individual_string = try allocator.alloc(u8, string.len);
        @memcpy(individual_string, string);

        return Individual{
            .string = individual_string,
            .fitness = fitness orelse 0,
            // std.math.maxInt(u32),
        };
    }

    // The distance is calculated based on the difference in position
    // between the characters in the GENES string and the target string.
    fn calculateFitness(self: *Individual, target: []const u8) void {
        // self.fitness = 0;
        // for (self.string, 0..) |char, i| {
        //     if (char != target[i]) {
        //         self.fitness += 1;
        //     }
        // }
        // return self.*;
        self.fitness = 0;
        for (self.string, 0..) |char, i| {
            if (char != target[i]) {
                // Penalize more for being far from the correct character
                // This creates a more informative gradient for selection
                const target_char_index = std.mem.indexOfScalar(u8, GENES, target[i]) orelse GENES.len;
                const individual_char_index = std.mem.indexOfScalar(u8, GENES, char) orelse GENES.len;

                // Calculate distance between characters in the GENES string
                const distance = @abs(@as(i32, @intCast(target_char_index)) - @as(i32, @intCast(individual_char_index)));

                // Fitness is increased (worse) by the distance
                self.fitness += distance + 1;
            }
        }
    }

    fn deinit(self: *Individual, allocator: std.mem.Allocator) void {
        allocator.free(self.string);
    }

    fn mate(
        self: *const Individual,
        allocator: std.mem.Allocator,
        parent2: *const Individual,
        ran: Random,
        target: []const u8,
    ) !Individual {
        const child_string = try allocator.alloc(u8, target.len);
        for (child_string, 0..) |*gene, i| {
            const p = ran.float(f32);
            if (p < GENE_MIX) {
                gene.* = self.string[i];
            } else if (p < GENE_MIX * 2) {
                gene.* = parent2.string[i];
            } else {
                gene.* = GENES[ran.intRangeAtMost(usize, 0, GENES.len - 1)];
            }
        }
        return Individual{
            .string = child_string,
            .fitness = blk: {
                var fitness: usize = 0;
                for (child_string, 0..) |char, i| {
                    if (char != target[i]) {
                        const target_char_index = std.mem.indexOfScalar(u8, GENES, target[i]) orelse GENES.len;
                        const individual_char_index = std.mem.indexOfScalar(u8, GENES, char) orelse GENES.len;

                        const distance = @abs(@as(i32, @intCast(target_char_index)) - @as(i32, @intCast(individual_char_index)));

                        fitness += distance + 1;
                    }
                }
                break :blk fitness;
            },
        };
    }
};

fn createGnome(allocator: std.mem.Allocator, len: usize, ran: Random) ![]u8 {
    const gnome = try allocator.alloc(u8, len);
    for (gnome) |*gene| {
        gene.* = GENES[
            ran.intRangeAtMost(
                usize,
                0,
                GENES.len - 1,
            )
        ];
    }
    return gnome;
}

fn individualLessThan(context: void, a: Individual, b: Individual) bool {
    _ = context;
    return a.fitness < b.fitness;
}

fn tournamentSelection(population: []Individual, tournament_size: usize, ran: Random) *const Individual {
    var best_individual = &population[
        ran.intRangeAtMost(
            usize,
            0,
            population.len - 1,
        )
    ];

    for (1..tournament_size) |_| {
        const candidate = &population[
            ran.intRangeAtMost(
                usize,
                0,
                population.len - 1,
            )
        ];
        if (candidate.fitness < best_individual.fitness) {
            best_individual = candidate;
        }
    }

    return best_individual;
}

const RunningStats = struct {
    mean: f64,
    std_dev: f64,
    min: usize,
    max: usize,

    fn init() RunningStats {
        return RunningStats{
            .mean = 0,
            .std_dev = 0,
            .min = std.math.maxInt(u32),
            .max = 0,
        };
    }

    fn calc_mean(self: *RunningStats, counts: []usize) void {
        var sum: f64 = 0;
        for (counts) |count| {
            sum += @as(f64, @floatFromInt(count));
        }
        self.mean = sum / @as(f64, @floatFromInt(counts.len));
    }
    fn calc_std_dev_min_max(self: *RunningStats, counts: []usize) void {
        if (counts.len < 2) return;
        if (self.min == 0) self.min = counts[0];
        if (self.max == 0) self.max = counts[0];
        var sum: f64 = 0;

        for (counts) |count| {
            if (count > self.max) {
                self.max = count;
            }
            if (count < self.min) {
                self.min = count;
            }
            sum += std.math.pow(f64, (@as(f64, @floatFromInt(count)) - self.mean), 2);
        }
        self.std_dev = std.math.sqrt(sum / @as(f64, @floatFromInt(counts.len - 1)));
    }

    // mutate the Individual struct
    fn calc(self: *RunningStats, counts: []usize) void {
        self.calc_mean(counts);
        self.calc_std_dev_min_max(counts);
    }
};

const MetaResult = struct {
    elitism: usize,
    mean: f64,
    std_dev: f64,
    min: usize,
    max: usize,
};

fn runGeneticAlgorithm(
    allocator: std.mem.Allocator,
    elitism: usize,
    ran: Random,
    input: []const u8,
) !usize {
    var population = std.ArrayList(Individual).init(allocator);
    defer {
        for (population.items) |*individual| {
            individual.deinit(allocator);
        }
        population.deinit();
    }

    var new_generation = std.ArrayList(Individual).init(allocator);
    defer {
        for (new_generation.items) |*individual| {
            individual.deinit(allocator);
        }
        new_generation.deinit();
    }

    // Create initial population
    for (0..POPULATION_SIZE) |_| {
        const gnome = try createGnome(
            allocator,
            input.len,
            ran,
        );
        defer allocator.free(gnome);

        var individual = try Individual.init(
            allocator,
            gnome,
            null,
        );

        individual.calculateFitness(input);
        try population.append(individual);
    }

    var generation: usize = 0;

    while (generation < MAX_GENERATIONS) : (generation += 1) {
        // Sort population by fitness
        std.mem.sort(
            Individual,
            population.items,
            {},
            individualLessThan,
        );

        // End condition: check if we've found the target
        if (population.items[0].fitness == 0) {
            return generation;
        }

        // Generate new population

        // Copy top individuals into new generation
        const elitism_count = @as(usize, (elitism * POPULATION_SIZE) / 100);
        for (population.items[0..elitism_count]) |individual| {
            try new_generation.append(try Individual.init(
                allocator,
                individual.string,
                individual.fitness,
            ));
        }

        // Mating: fill the rest of the new generation with an offsptring from a tournament
        const mating_count = POPULATION_SIZE - elitism_count;
        for (0..mating_count) |_| {
            // const parent1 = &population.items[ran.intRangeAtMost(usize, 0, 50)];
            // const parent2 = &population.items[ran.intRangeAtMost(usize, 0, 50)];
            const parent1 = tournamentSelection(
                population.items,
                TOURNAMENT_SIZE,
                ran,
            );
            const parent2 = tournamentSelection(
                population.items,
                TOURNAMENT_SIZE,
                ran,
            );
            const offspring = try parent1.mate(
                allocator,
                parent2,
                ran,
                input,
            );
            try new_generation.append(offspring);
        }

        for (population.items) |*individual| {
            individual.deinit(allocator);
        }

        // Replace old population
        std.mem.swap(ArrayList(Individual), &population, &new_generation);
        new_generation.clearRetainingCapacity();
    }
    return MAX_GENERATIONS;
}

/// nif: run/2 Threaded
pub fn run(elitism: usize, input: []const u8) !MetaResult {
    const num_cpus = Thread.getCpuCount() catch 2;

    var thread_pool = try ThreadPool.init(.{
        .allocator = beam_allocator,
        .num_threads = num_cpus,
        .elitism = elitism,
    });

    defer thread_pool.deinit();

    try thread_pool.runParallelTrials(input);

    var stats = RunningStats.init();
    stats.calc(thread_pool.results);

    const meta_array = MetaResult{
        .elitism = elitism,
        .mean = stats.mean,
        .std_dev = stats.std_dev,
        .min = stats.min,
        .max = stats.max,
    };

    // print("MetaResult: elitism={}, mean={}, std_dev={}, min={}, max={}\n", .{
    //     meta_array.elitism,
    //     meta_array.mean,
    //     meta_array.std_dev,
    //     meta_array.min,
    //     meta_array.max,
    // });

    return .{
        .elitism = meta_array.elitism,
        .mean = meta_array.mean,
        .std_dev = meta_array.std_dev,
        .min = meta_array.min,
        .max = meta_array.max,
    };
}
