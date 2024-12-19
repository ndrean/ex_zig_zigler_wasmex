defmodule GAT do
  use Zig,
    otp_app: :ex_genetic_zig,
    zig_code_path: "ga_threaded.zig",
    # leak_check: true,
    release_mode: :fast


  def calc(elitism_rate, input) do
    run(elitism_rate, input)
  end

  def start do
    :nok
  end
end

defmodule GA do
  use Zig,
    otp_app: :ex_genetic_zig,
    zig_code_path: "ga_zigler.zig",
    release_mode: :fast

  def calc(elitism_rate, input) do
    run(elitism_rate, input)
  end
end
