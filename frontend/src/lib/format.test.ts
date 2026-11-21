import { formatBytes, formatNumber, formatSeconds, scopeLabel } from "./format";

describe("formatters", () => {
  it("formats numbers, bytes and durations", () => {
    expect(formatNumber(0.12345)).toBe("0.123");
    expect(formatNumber(null)).toBe("—");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(874241)).toBe("853.8 KB");
    expect(formatSeconds(0.25)).toBe("250 ms");
    expect(formatSeconds(12.3)).toBe("12.3 s");
  });

  it("labels evaluation scopes", () => {
    expect(scopeLabel("internal")).toBe("Interna (teste)");
    expect(scopeLabel("external:client-02")).toBe("Externa: client-02");
  });
});
