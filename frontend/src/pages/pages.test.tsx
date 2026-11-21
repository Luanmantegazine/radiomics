import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { api } from "../api/client";
import { fixtures } from "../test/fixtures";
import { ExperimentPage } from "./ExperimentPage";
import { ExperimentsPage } from "./ExperimentsPage";
import { NewExperimentPage, parseConfig } from "./NewExperimentPage";

vi.mock("../components/Plot", () => ({
  Plot: ({ filename }: { filename: string }) => <div data-testid={`plot-${filename}`} />,
  PALETTE: ["#000", "#111", "#222", "#333"],
}));

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      experiments: vi.fn(), summary: vi.fn(), experiment: vi.fn(), templates: vi.fn(),
      validate: vi.fn(), create: vi.fn(), run: vi.fn(), evaluation: vi.fn(),
    },
  };
});
const mocked = vi.mocked(api);

function renderAt(path: string, element: JSX.Element, route = path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path={route} element={element} />
        <Route path="/experiments/:id" element={<div>experiment page</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

afterEach(() => vi.clearAllMocks());

describe("experiments list", () => {
  it("lists experiments with status and data origin", async () => {
    mocked.experiments.mockResolvedValue([fixtures.summary.experiment]);
    renderAt("/", <ExperimentsPage />);
    expect(await screen.findByText("stage6-harmonization-evaluation")).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("synthetic")).toBeInTheDocument();
  });
});

describe("experiment page", () => {
  it("warns about synthetic data and shows only the available result tabs", async () => {
    mocked.summary.mockResolvedValue({
      ...fixtures.summary,
      available_results: { harmonization: false, feature_selection: true, models: true,
                           evaluation: false },
    });
    mocked.experiment.mockResolvedValue(fixtures.detail);
    renderAt(`/experiments/${fixtures.detail.id}`, <ExperimentPage />, "/experiments/:id");
    expect(await screen.findByText(/Dados sintéticos\./)).toBeInTheDocument();
    const tabs = within(screen.getByRole("tablist")).getAllByRole("tab").map((t) => t.textContent);
    expect(tabs).toEqual(["Visão geral", "Sites e rodadas", "Seleção", "Modelos", "Custos",
                          "Exportar"]);
  });

  it("shows the evaluation tab with pooled metrics", async () => {
    mocked.summary.mockResolvedValue(fixtures.summary);
    mocked.experiment.mockResolvedValue(fixtures.detail);
    mocked.evaluation.mockResolvedValue(fixtures.evaluation);
    renderAt(`/experiments/${fixtures.detail.id}?tab=evaluation`, <ExperimentPage />,
      "/experiments/:id");
    expect(await screen.findByText("Métricas por modelo e horizonte")).toBeInTheDocument();
    expect(screen.getByTestId(/plot-.*validacao-externa/)).toBeInTheDocument();
    const auc = fixtures.evaluation.evaluations[0].pooled[0].auc!;
    expect(screen.getAllByText(auc.toFixed(3)).length).toBeGreaterThan(0);
  });
});

describe("new experiment", () => {
  it("parses YAML and JSON and rejects non-objects", () => {
    expect(parseConfig("seed: 3\nname: x").config).toEqual({ seed: 3, name: "x" });
    expect(parseConfig('{"seed": 3}').config).toEqual({ seed: 3 });
    expect(parseConfig("- a\n- b").error).toMatch(/objeto/);
    expect(parseConfig("seed: [1, 2").error).toMatch(/sintaxe/);
  });

  it("shows validation errors from the backend", async () => {
    mocked.templates.mockResolvedValue(fixtures.templates);
    mocked.validate.mockResolvedValue({
      valid: false, errors: [{ location: "seed", message: "must be >= 0" }],
    } as never);
    renderAt("/new", <NewExperimentPage />);
    await userEvent.click(await screen.findByText("stage1-infrastructure-check"));
    await userEvent.click(screen.getByRole("button", { name: "Validar" }));
    expect(await screen.findByText("Configuração inválida")).toBeInTheDocument();
    expect(screen.getByText("seed")).toBeInTheDocument();
    expect(mocked.create).not.toHaveBeenCalled();
  });

  it("creates and runs a valid configuration from a template", async () => {
    mocked.templates.mockResolvedValue(fixtures.templates);
    mocked.validate.mockResolvedValue({ valid: true, errors: [], config_hash: "abc",
                                        phases: ["infrastructure"], expected_clients: [] } as never);
    mocked.create.mockResolvedValue({ ...fixtures.detail, id: "exp-new" });
    mocked.run.mockResolvedValue(fixtures.summary.experiment);
    renderAt("/new", <NewExperimentPage />);
    await userEvent.click(await screen.findByText("stage1-infrastructure-check"));
    await userEvent.click(screen.getByRole("button", { name: "Criar e executar" }));
    await waitFor(() => expect(mocked.run).toHaveBeenCalledWith("exp-new"));
    const sent = mocked.create.mock.calls[0][0] as Record<string, unknown>;
    expect(sent.name).toBe("stage1-infrastructure-check");
    expect(await screen.findByText("experiment page")).toBeInTheDocument();
  });
});
