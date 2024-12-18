// # ~Z"""
// #   const beam = @import("beam");

// #   pub fn double_atom(string: []u8) !beam.term {
// #       var double_string = try beam.allocator.alloc(u8, string.len * 2);
// #       defer beam.allocator.free(double_string);

// #       for (string, 0..) |char, i| {
// #           double_string[i] = char;
// #           double_string[i + string.len] = char;
// #       }

// #       return beam.make_into_atom(double_string, .{});
// #   }

// #   pub fn string_count(string: []u8) i64 {
// #       return @intCast(string.len);
// #   }
// # """
