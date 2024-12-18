const std = @import("std");
const math = @import("std").math;
const ArrayList = std.ArrayList;
const Random = std.Random;
const native_endian = @import("builtin").target.cpu.arch.endian();

const POPULATION_SIZE: comptime_int = 100;
const GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!\"#%&/()=?@${[]}";
const MAX_TARGET_LENGTH: comptime_int = 50;
const MAX_GENERATIONS: comptime_int = 5_000; // Prevent infinite loops
const MAX_TRIALS: comptime_int = 32;
const GENE_MIX: comptime_float = 0.45;
const TOURNAMENT_SIZE = 10;

var target: []u8 = undefined;

var wasm_allocator = std.heap.wasm_allocator;

const Individual = struct {
    string: []u8,
    fitness: usize,

    fn init(allocator: std.mem.Allocator, string: []const u8) !Individual {
        const individual_string = try allocator.alloc(u8, string.len);
        @memcpy(individual_string, string);

        var individual = Individual{
            .string = individual_string,
            .fitness = 0,
        };
        return individual.calculateFitness();
    }

    fn deinit(self: *Individual, allocator: std.mem.Allocator) void {
        allocator.free(self.string);
    }

    // mutate the Individual struct and compute the fitness:
    // it is the number of differences between the genes and the target
    fn calculateFitness(self: *Individual) Individual {
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

        return self.*;
    }

    fn mate(
        self: *const Individual,
        allocator: std.mem.Allocator,
        parent2: *const Individual,
        ran: Random,
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
        // return try Individual.init(allocator, child_string);
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
        gene.* = GENES[ran.intRangeAtMost(usize, 0, GENES.len - 1)];
    }
    return gnome;
}

fn individualLessThan(context: void, a: Individual, b: Individual) bool {
    _ = context;
    return a.fitness < b.fitness;
}

fn tournamentSelection(population: []Individual, tournament_size: usize, ran: Random) *const Individual {
    var best_individual = &population[ran.intRangeAtMost(usize, 0, population.len - 1)];

    for (1..tournament_size) |_| {
        const candidate = &population[ran.intRangeAtMost(usize, 0, population.len - 1)];
        if (candidate.fitness < best_individual.fitness) {
            best_individual = candidate;
        }
    }

    return best_individual;
}

fn runGeneticAlgorithm(
    allocator: std.mem.Allocator,
    elitism: usize,
    ran: Random,
) !usize {
    var population = ArrayList(Individual).init(allocator);

    defer {
        for (population.items) |*individual| {
            individual.deinit(allocator);
            // allocator.free(individual.string);
        }
        population.deinit();
    }
    var new_generation = ArrayList(Individual).init(allocator);
    defer {
        for (new_generation.items) |*individual| {
            individual.deinit(allocator);
        }
        new_generation.deinit();
    }
    // Create initial population
    for (0..POPULATION_SIZE) |_| {
        const gnome = try createGnome(allocator, target.len, ran);
        defer allocator.free(gnome);
        const individual = try Individual.init(allocator, gnome);
        try population.append(individual);
    }

    var generation: usize = 0;

    while (generation < MAX_GENERATIONS) : (generation += 1) {
        // Sort population by fitness
        std.mem.sort(Individual, population.items, {}, individualLessThan);

        // Check if we've found the target
        if (population.items[0].fitness == 0) {
            return generation;
        }

        // Generate new population

        // Elitism
        const elitism_count = @as(usize, (elitism * POPULATION_SIZE) / 100);
        for (population.items[0..elitism_count]) |individual| {
            try new_generation.append(try Individual.init(allocator, individual.string));
        }

        // Mating
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
            const offspring = try parent1.mate(allocator, parent2, ran);
            try new_generation.append(offspring);
        }

        // Replace old population
        std.mem.swap(ArrayList(Individual), &population, &new_generation);
        for (new_generation.items) |*individual| {
            individual.deinit(allocator);
        }

        new_generation.clearRetainingCapacity();
    }
    // If solution not found
    return MAX_GENERATIONS;
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

// Freeing the allocated buffer
export fn free(ptr: [*]u8, len: usize) void {
    wasm_allocator.free(ptr[0..len]);
}

const MetaResult = struct {
    elitism: usize,
    mean: f64,
    std_dev: f64,
    min: usize,
    max: usize,
    len: usize,
    // target_length: usize,
    // target_first_bytes: [8]u8 = [_]u8{0} ** 8,
};

fn serialise(meta_result: *MetaResult, allocator: std.mem.Allocator) ![]u8 {
    const buffer = allocator.alloc(u8, 8 * 6) catch {
        @panic("Could not allocate buffer");
    };
    errdefer allocator.free(buffer);
    var stream = std.io.fixedBufferStream(buffer);
    const writer = stream.writer();
    try writer.writeInt(u64, @as(u64, @intCast(meta_result.elitism)), native_endian);
    try writer.writeAll(std.mem.asBytes(&meta_result.mean));
    try writer.writeAll(std.mem.asBytes(&meta_result.std_dev)); // f64
    try writer.writeInt(u64, @as(u64, @intCast(meta_result.min)), .little);
    try writer.writeInt(u64, @as(u64, @intCast(meta_result.max)), native_endian);
    try writer.writeInt(u64, @as(u64, @intCast(meta_result.len)), native_endian);

    // try writer.writeInt(u64, @as(u64, @intCast(meta_result.target_length)), native_endian);
    // try writer.writeAll(meta_result.target_first_bytes[0..8]);

    return buffer;
}

export fn alloc(len: usize) ?[*]u8 {
    if (len > MAX_TARGET_LENGTH) {
        return null;
    }
    target = wasm_allocator.alloc(u8, len) catch {
        return null;
    };
    return target.ptr;
}

// debug function
export fn get_target() ?[*]u8 {
    return target.ptr;
}

export fn set_target(ptr: [*]u8, len: usize) void {
    @memcpy(target, ptr[0..len]);
}

export fn run(elitism: usize) [*]u8 {
    var rand = Random.DefaultPrng.init(12345);
    const ran = rand.random();
    var generations = std.ArrayList(usize).init(wasm_allocator);
    defer generations.deinit();
    defer wasm_allocator.free(target);

    for (0..MAX_TRIALS) |_| {
        const count = runGeneticAlgorithm(wasm_allocator, elitism, ran) catch {
            @panic("Could not run genetic algorithm");
        };
        generations.append(count) catch {
            @panic("Could not append count to generations");
        };
    }
    var stats = RunningStats.init();
    const len = generations.items.len;
    const results = generations.toOwnedSlice() catch {
        @panic("Could not convert generations to owned slice");
    };
    defer wasm_allocator.free(results); //
    stats.calc(results);

    var meta_array = MetaResult{
        .elitism = elitism,
        .mean = stats.mean,
        .std_dev = stats.std_dev,
        .min = stats.min,
        .max = stats.max,
        .len = len,
        // .target_length = target.len,
        // .target_first_bytes = blk: {
        //     var first_bytes: [8]u8 = undefined;
        //     @memcpy(&first_bytes, target[0..8]);
        //     break :blk first_bytes;
        // },
    };

    const serialized = serialise(&meta_array, wasm_allocator) catch {
        @panic("Could not serialise meta results");
    };
    return serialized.ptr;
}
