defmodule ExGeneticZig.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_genetic_zig,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {ExGeneticZig.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:zigler, "~> 0.13.3", runtime: false},
      {:wasmex, "~> 0.9.2"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
