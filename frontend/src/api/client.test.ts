import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from "axios";

import { errorMessage, setToken, withToken } from "./client";

const config = () => ({ headers: new AxiosHeaders() }) as InternalAxiosRequestConfig;

describe("api client", () => {
  afterEach(() => setToken(""));

  it("sends the bearer token only when one is configured", () => {
    expect(withToken(config()).headers.get("Authorization")).toBeUndefined();
    setToken("s3cret");
    expect(withToken(config()).headers.get("Authorization")).toBe("Bearer s3cret");
  });

  it("turns API errors into readable messages", () => {
    const unauthorized = new AxiosError("x", "401", undefined, undefined, {
      status: 401, data: {}, statusText: "", headers: {}, config: config(),
    });
    expect(errorMessage(unauthorized)).toMatch(/Token/);
    const notFound = new AxiosError("x", "404", undefined, undefined, {
      status: 404, data: { detail: "experiment not found" }, statusText: "", headers: {},
      config: config(),
    });
    expect(errorMessage(notFound)).toBe("experiment not found");
  });
});
