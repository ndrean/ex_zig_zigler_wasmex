const std = @import("std");
const math = @import("std").math;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Random = std.Random;
const ArenaAllocator = std.heap.ArenaAllocator;

const POPULATION_SIZE: comptime_int = 100;
const GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!\"#%&/()=?@${[]}";
const MAX_TARGET_LENGTH: comptime_int = 100;
const MAX_GENERATIONS: comptime_int = 10_000; // Prevent infinite loops
const MAX_TRIALS: comptime_int = 500;
const MAX_ELITISM: comptime_int = 75;
const ELITISM_STEP: comptime_int = 2;
const ELITISM_START: comptime_int = 2;
const TARGET = "I love you baby!";

const Individual = struct {
    string: []u8,
    fitness: usize,

    fn init(allocator: std.mem.Allocator, string: []const u8) !Individual {
        var self = Individual{
            .string = try allocator.dupe(u8, string),
            .fitness = 0,
        };
        self.calculateFitness();
        return self;
    }

    // mutate the Individual struct and compute the fitness:
    // it is the number of differences between the genes and the target
    fn calculateFitness(self: *Individual) void {
        self.fitness = 0;
        for (self.string, 0..) |char, i| {
            if (char != TARGET[i]) {
                self.fitness += 1;
            }
        }
    }

    fn mate(self: *const Individual, allocator: std.mem.Allocator, parent2: *const Individual, rng: Random) !Individual {
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

fn runGeneticAlgorithm(allocator: std.mem.Allocator, elitism: usize, ran: Random) !usize {
    var population = ArrayList(Individual).init(allocator);
    defer population.deinit();

    // Create initial population
    for (0..POPULATION_SIZE) |_| {
        const gnome = try createGnome(allocator, ran);
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
        var new_generation = ArrayList(Individual).init(allocator);

        // Elitism
        const elitism_count = @as(usize, (elitism * POPULATION_SIZE) / 100);
        for (population.items[0..elitism_count]) |individual| {
            try new_generation.append(try Individual.init(allocator, individual.string));
        }

        // Mating
        const mating_count = POPULATION_SIZE - elitism_count;
        for (0..mating_count) |_| {
            const parent1 = &population.items[ran.intRangeAtMost(usize, 0, 50)];
            const parent2 = &population.items[ran.intRangeAtMost(usize, 0, 50)];
            const offspring = try parent1.mate(allocator, parent2, ran);
            try new_generation.append(offspring);
        }

        // Replace old population
        population.deinit();
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
            .min = 0,
            .max = 0,
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

// fn generateRandomInterface(seed: u64) Random {
//     // var rand = Random.DefaultPrng.init(@intCast(std.time.timestamp()));
//     var rand = Random.DefaultPrng.init(seed);
//     // we generate a random number interface: each time it is used, a new random number is generated
//     return rand.random();
// }
pub fn main() !void {
    var arena = ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // we seed with an i64 casted to a u64 to avoid a compiler error, at compile time
    // var rand = Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    // we generate a random number interface: each time it is used, a new random number is generated
    // const ran = rand.random();
    // const ran = generateRandomInterface();

    const file = try std.fs.cwd().createFile("genetic_algorithm_results.csv", .{});
    defer file.close();
    const writer = file.writer();

    var elitism: usize = ELITISM_START;

    try writer.writeAll("Elitism,Avg,StdDev,Min,Max\n");

    while (elitism <= MAX_ELITISM) : (elitism += ELITISM_STEP) {
        var stats = RunningStats.init();
        var rand = Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const ran = rand.random();
        // const seed = @as(u64, @intCast(std.time.timestamp()));
        // const ran = generateRandomInterface(seed);

        for (0..MAX_TRIALS) |_| {
            const generations = try runGeneticAlgorithm(allocator, elitism, ran);
            stats.update(generations);
        }

        // Write summary statistics
        try writer.print("{d},{d:.2},{d:.2},{d},{d}\n", .{ elitism, stats.mean, stats.getStandardDeviation(), stats.min, stats.max });

        std.debug.print("Elitism: {d}, Avg: {d:.2}, StdDev: {d:.2}, min: {d}, max: {d}\n", .{ elitism, stats.mean, stats.getStandardDeviation(), stats.min, stats.max });
    }
    // Now print all the accumulated results at once
    std.debug.print("The results have been saved to 'genetic_algorithm_results.csv'\n", .{});
}
