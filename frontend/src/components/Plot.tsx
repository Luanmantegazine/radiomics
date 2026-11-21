import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const PlotlyComponent = createPlotlyComponent(Plotly);

export const PALETTE = ["#2f6fb0", "#d1712b", "#3a9a5b", "#8c5bb5", "#c2416b", "#5f7487",
  "#b5952f", "#2b9aa0"];

interface Props {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height?: number;
  filename: string;
}

const AXIS = { gridcolor: "#e6eaef", zerolinecolor: "#c9d1da", automargin: true };

/** Plotly chart with consistent styling and PNG export of the figure. */
export function Plot({ data, layout, height = 340, filename }: Props) {
  return (
    <PlotlyComponent
      data={data}
      layout={{
        height,
        autosize: true,
        margin: { l: 56, r: 16, t: 36, b: 48 },
        font: { family: "Roboto, system-ui, sans-serif", size: 12, color: "#1d2733" },
        colorway: PALETTE,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        // Legend above the plot area, so it never collides with axis titles.
        legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom" },
        ...layout,
        xaxis: { ...AXIS, ...layout?.xaxis },
        yaxis: { ...AXIS, ...layout?.yaxis },
      }}
      config={{
        displaylogo: false,
        responsive: true,
        toImageButtonOptions: { format: "png", filename, scale: 2 },
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
      }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
