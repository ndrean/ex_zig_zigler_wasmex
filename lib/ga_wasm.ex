defmodule GW do

  def calc(elitism, input) do
    bin = File.read!("./zig/zig-out/bin/ga.wasm")
    len = String.length(input)

    # # Task.async_stream(2..75//2,
    # # fn elitism ->

    {:ok, pid} = Wasmex.start_link(%{bytes: bin, wasi: true})

    {:ok, store} = Wasmex.store(pid)
    {:ok, memory} = Wasmex.memory(pid)

    {:ok, [input_idx]} = Wasmex.call_function(pid, "alloc", [len])
    :ok = Wasmex.Memory.write_binary(store, memory, input_idx, input)
    {:ok, _} = Wasmex.call_function(pid, "set_target", [input_idx, len])

    {:ok, [index]} = Wasmex.call_function(pid, "run", [elitism], 20_000)
    response = Wasmex.Memory.read_binary(store, memory, index, 8 * 6)
    Wasmex.call_function(pid, "free", [index, 8 * 6])

    <<elitism::unsigned-little-64, mean::float-little-64, std_dev::float-little-64,
      min::unsigned-little-64, max::unsigned-little-64, len::unsigned-little-64>> = response
      # target_length::unsigned-little-64, target_first_bytes::binary-size(8)>> = response
    %{
      elitism: elitism,
      mean: mean,
      std_dev: std_dev,
      min: min,
      max: max,
      trials: len,
      # target_length: target_length,
      # target_first_bytes: target_first_bytes
    }
  rescue
    e in MatchError ->
      IO.inspect(e, label: "Received response")
      IO.puts("Received response size: #{byte_size(e)}")

      raise "Pattern matching failed"
    # end,
    # timeout: :infinity
    # )
    # |> Stream.map(fn {:ok, result} -> result end)
    # |> Enum.to_list()
  end
end
