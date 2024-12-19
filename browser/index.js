export async function initializeWasmModule() {
  const response = await fetch("ga_browser.wasm");
  const binary = await response.arrayBuffer();

  const memory = new WebAssembly.Memory({ initial: 512 });
  const { instance } = await WebAssembly.instantiate(binary, {
    env: { memory },
  });

  return {
    setInput: (input) => {
      const { memory, set_target, alloc } = instance.exports;

      const len = input.length;
      const stringBuffer = new TextEncoder("ascii").encode(input); // get an ArrayBuffer of the input string

      const ptr = alloc(len); // get an index from the Zig alcator
      if (ptr < memory.buffer.byteLength && ptr > 0) {
        const wasmMemory = new Uint8Array(memory.buffer);
        wasmMemory.set(stringBuffer, ptr); // copy the ArrayBuffer to the wasm memory

        set_target(ptr, len); // Zig set_target
      } else {
        console.error("Failed to allocate memory");
      }
    },

    run: (elitism = 10) => {
      const { run, memory, free } = instance.exports;

      const resultPtr = run(elitism);
      const view = new DataView(memory.buffer, resultPtr, 8 * 6);

      const result = {
        elitism: Number(view.getBigUint64(0, true)),
        mean: view.getFloat64(8, true),
        stdDev: view.getFloat64(16, true),
        min: Number(view.getBigUint64(24, true)),
        max: Number(view.getBigUint64(32, true)),
        len: Number(view.getBigUint64(40, true)),
        // targetLength: Number(view.getBigUint64(48, true)),
        // targetFirstBytes: Array.from(
        //   new Uint8Array(memory.buffer, resultPtr + 7 * 8, 8)
        // )
        //   .map((b) => String.fromCharCode(b))
        //   .join(""),
      };

      free(resultPtr, 64);
      return result;
    },
  };
}
