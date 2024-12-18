const std = @import("std");
const math = @import("std").math;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Random = std.Random;
const ArenaAllocator = std.heap.ArenaAllocator;

const POPULATION_SIZE: comptime_int = 100;
const GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!\"#%&/()=?@${[]}";
const MAX_TARGET_LENGTH: comptime_int = 100;
const MAX_GENERATIONS: comptime_int = 1_000; // Prevent infinite loops
const MAX_TRIALS: comptime_int = 10;
const MAX_ELITISM: comptime_int = 75;
const ELITISM_STEP: comptime_int = 1;
const ELITISM_START: comptime_int = 2;
const GENE_MIX: comptime_float = 0.45;

const TARGET = "I love you baby!";

const Individual = struct {
    string: []u8,
    fitness: usize,

    fn init(allocator: std.mem.Allocator, string: []const u8) !Individual {
        var individual = Individual{
            .string = try allocator.dupe(u8, string),
            .fitness = 0,
        };
        return individual.calculateFitness();
    }

    // mutate the Individual struct and compute the fitness:
    // it is the number of differences between the genes and the target
    fn calculateFitness(self: *Individual) Individual {
        self.fitness = 0;
        for (self.string, 0..) |char, i| {
            if (char != TARGET[i]) {
                self.fitness += 1;
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
        const child_string = try allocator.alloc(u8, TARGET.len);
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
        return try Individual.init(allocator, child_string);
    }
};

fn createGnome(allocator: std.mem.Allocator, ran: Random) ![]u8 {
    const gnome = try allocator.alloc(u8, TARGET.len);
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

fn runGeneticAlgorithm(allocator: std.mem.Allocator, elitism: usize, ran: Random) !usize {
    var population = ArrayList(Individual).init(allocator);
    defer {
        for (population.items) |*ind| allocator.free(ind.string);
        population.deinit();
    }

    // Create initial population
    for (0..POPULATION_SIZE) |_| {
        const gnome = try createGnome(allocator, ran);
        const individual = try Individual.init(allocator, gnome);
        try population.append(individual);
    }

    var generation: usize = 0;
    // <-------- Early stop condition
    // var generations_without_improvement: usize = 0;
    // var best_fitness: usize = population.items[0].fitness;
    // <-------- End of early stop condition

    while (generation < MAX_GENERATIONS) : (generation += 1) {
        // Sort population by fitness
        std.mem.sort(Individual, population.items, {}, individualLessThan);

        // Check if we've found the target
        if (population.items[0].fitness == 0) {
            return generation;
        }

        // <-------- Early stop condition
        // const current_best_fitness = population.items[0].fitness;
        // const improvement_threshold = 5; // Only count significant improvements

        // if (best_fitness > current_best_fitness and
        //     (best_fitness - current_best_fitness) >= improvement_threshold)
        // {
        //     best_fitness = current_best_fitness;
        //     generations_without_improvement = 0;
        // } else {
        //     generations_without_improvement += 1;
        // }

        // // More forgiving early stop condition
        // // Allow up to 25% of max generations without significant improvement
        // if (generations_without_improvement > MAX_GENERATIONS / 4) {
        //     // If we're close to solution, continue
        //     if (best_fitness <= TARGET.len / 4) {
        //         continue;
        //     }
        //     break;
        // }
        // <-------- End of early stop condition

        // Generate new population
        var new_generation = ArrayList(Individual).init(allocator);

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
            const parent1 = tournamentSelection(population.items, 3, ran);
            const parent2 = tournamentSelection(population.items, 3, ran);
            const offspring = try parent1.mate(allocator, parent2, ran);
            try new_generation.append(offspring);
        }

        // Replace old population
        for (population.items) |*ind| allocator.free(ind.string);
        population = new_generation;
    }

    return MAX_GENERATIONS; // If solution not found
}

const RunningStats = struct {
    count: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,

    fn init() RunningStats {
        return RunningStats{
            .count = 0,
            .mean = 0,
            .m2 = 0,
            .min = std.math.inf(f64),
            .max = -std.math.inf(f64),
        };
    }

    // mutate the Individual struct
    fn update(self: *RunningStats, value: usize) void {
        self.count += 1;
        const delta = @as(f64, @floatFromInt(value)) - self.mean;
        self.mean += delta / @as(f64, @floatFromInt(self.count));
        const delta2 = @as(f64, @floatFromInt(value)) - self.mean;
        self.m2 += delta * delta2;
        self.mixMax(value);
    }

    fn getStandardDeviation(self: *const RunningStats) f64 {
        if (self.count < 2) return 0;
        return std.math.sqrt(self.m2 / @as(f64, @floatFromInt(self.count - 1)));
    }

    fn mixMax(self: *RunningStats, value: usize) void {
        const v = @as(f64, @floatFromInt(value));
        if (self.min == 0) self.min = v;
        if (v > self.max) {
            self.max = v;
        }
        if (v < self.min) {
            self.min = v;
        }
    }
};

pub fn main() !void {
    var arena = ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var rand = Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const ran = rand.random();

    const file = try std.fs.cwd().createFile("genetic_algorithm_results2.csv", .{});
    defer file.close();
    const writer = file.writer();

    // Write CSV header
    try writer.writeAll("Elitism,Avg,StdDev,Min,Max\n");

    var elitism: usize = ELITISM_START;

    while (elitism <= MAX_ELITISM) : (elitism += ELITISM_STEP) {
        var stats = RunningStats.init();

        for (0..MAX_TRIALS) |_| {
            const generations = try runGeneticAlgorithm(allocator, elitism, ran);
            stats.update(generations);

            // Free memory after each trial
            arena.deinit();
            arena = ArenaAllocator.init(std.heap.page_allocator);
        }

        // Write summary statistics to CSV
        try writer.print("{d},{d:.2},{d:.2},{d},{d}\n", .{ elitism, stats.mean, stats.getStandardDeviation(), stats.min, stats.max });

        // Print to console for immediate feedback
        std.debug.print("Elitism: {d}, Avg: {d:.2}, StdDev: {d:.2}, min: {d}, max: {d}\n", .{ elitism, stats.mean, stats.getStandardDeviation(), stats.min, stats.max });
    }

    // Ensure all data is written to the file
    try file.sync();

    std.debug.print("The results have been saved to 'genetic_algorithm_results.csv'\n", .{});
}
