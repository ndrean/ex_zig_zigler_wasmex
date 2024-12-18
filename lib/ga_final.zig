const std = @import("std");
const math = @import("std").math;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;

const POPULATION_SIZE: usize = 100;
const GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!\"#%&/()=?@${[]}";
const MAX_TARGET_LENGTH: comptime_int = 100;
const MAX_GENERATIONS: comptime_int = 10000; // Prevent infinite loops
const MAX_TRIALS: comptime_int = 40;
const MAX_ELITISM: comptime_int = 2;
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

    fn calculateFitness(self: *Individual) Individual {
        self.fitness = 0;
        for (self.string, 0..) |gene, i| {
            if (gene != TARGET[i]) {
                self.fitness += 1;
            }
        }
        return self.*;
    }

    fn mate(self: *const Individual, allocator: std.mem.Allocator, parent2: *const Individual, rng: std.Random) !Individual {
        const child_string = try allocator.alloc(u8, TARGET.len);
        for (child_string, 0..) |*gene, i| {
            const p = rng.float(f32);
            if (p < 0.45) {
                gene.* = self.string[i];
            } else if (p < 0.9) {
                gene.* = parent2.string[i];
            } else {
                gene.* = GENES[rng.intRangeAtMost(usize, 0, GENES.len - 1)];
            }
        }
        return try Individual.init(allocator, child_string);
    }
};

fn createGnome(allocator: std.mem.Allocator, rng: std.Random) ![]u8 {
    const gnome = try allocator.alloc(u8, TARGET.len);
    for (gnome) |*gene| {
        gene.* = GENES[rng.intRangeAtMost(usize, 0, GENES.len - 1)];
    }
    return gnome;
}

fn fitnessLessThan(context: void, a: Individual, b: Individual) bool {
    _ = context;
    return a.fitness < b.fitness;
}

// var TARGET: ArrayList(u8) = undefined;

fn runGeneticAlgorithm(allocator: std.mem.Allocator, rng: std.Random, elitism: usize) !usize {
    // we need an ArrayList as we have a dynamic array
    var population = ArrayList(Individual).init(allocator);
    defer {
        for (population.items) |*ind| allocator.free(ind.string);
        population.deinit();
    }

    // Create initial population
    for (0..POPULATION_SIZE) |_| {
        const gnome = try createGnome(allocator, rng);
        defer allocator.free(gnome);
        const individual = try Individual.init(allocator, gnome);
        try population.append(individual);
    }

    var generation: usize = 0;
    while (generation < MAX_GENERATIONS) : (generation += 1) {
        // Sort population by fitness
        std.mem.sort(Individual, population.items, {}, fitnessLessThan);

        // Check if we've found the target
        if (population.items[0].fitness == 0) {
            return generation;
        }

        // Generate new population
        var new_generation = ArrayList(Individual).init(allocator);
        // defer {
        //     for (new_generation.items) |*ind| allocator.free(ind.string);
        //     new_generation.deinit();
        // }

        // Elitism: keep strategy
        const elitism_count = @as(usize, (elitism * POPULATION_SIZE) / 100);
        for (population.items[0..elitism_count]) |individual| {
            try new_generation.append(try Individual.init(allocator, individual.string));
        }

        // Mating:
        const mating_count = POPULATION_SIZE - elitism_count;
        for (0..mating_count) |_| {
            const parent1 = &population.items[rng.intRangeAtMost(usize, 0, 50)];
            const parent2 = &population.items[rng.intRangeAtMost(usize, 0, 50)];
            const offspring = try parent1.mate(allocator, parent2, rng);
            try new_generation.append(offspring);
        }

        // Replace old population
        // population.deinit();
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
            .mean = 0.0,
            .m2 = 0.0,
            .min = 0, //std.math.inf(f64),
            .max = 0, //-std.math.inf(f64),
        };
    }

    fn update(self: *RunningStats, value: usize) void {
        self.count += 1;
        const delta: f64 = @as(f64, @floatFromInt(value)) - @as(f64, self.mean);
        self.mean += delta / @as(f64, @floatFromInt(self.count));
        self.m2 += delta * (@as(f64, @floatFromInt(value)) - self.mean);
        self.minMax(value);
        // self.min = if (self.min < @as(f64, @floatFromInt(value))) self.min else @as(f64, @floatFromInt(value));
        // self.max = if (self.max > @as(f64, @floatFromInt(value))) self.max else @as(f64, @floatFromInt(value));
    }

    fn getStandardDeviation(self: *RunningStats) f64 {
        if (self.count < 2) return 0;
        return math.sqrt(self.m2 / @as(f64, @floatFromInt(self.count - 1)));
    }

    fn minMax(self: *RunningStats, value: usize) void {
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

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rand = prng.random();

    const file = try std.fs.cwd().createFile("genetic_algorithm_results2.csv", .{});
    defer file.close();
    const writer = file.writer();

    // CSV header
    try writer.writeAll("Elitism,Avg,StdDev,Min,Max\n");

    var elitism: usize = ELITISM_START;

    while (elitism <= MAX_ELITISM) : (elitism += ELITISM_STEP) {
        var stats = RunningStats.init();

        for (0..MAX_TRIALS) |_| {
            const generations = try runGeneticAlgorithm(allocator, rand, elitism);
            stats.update(generations);
        }

        // Format into the temporary buffer
        try writer.print("{d},{d:.2},{d:.2},{d},{d}\n", .{ elitism, stats.mean, stats.getStandardDeviation(), stats.min, stats.max });
        print("Elitism: {d}, Avg: {d:.2}, StdDev: {d:.2}, Min: {d}, Max: {d}\n", .{ elitism, stats.mean, stats.getStandardDeviation(), stats.min, stats.max });
    }
    try file.sync();
    print("Results saved to 'genetic_algorithm_results.csv'\n", .{});
}
