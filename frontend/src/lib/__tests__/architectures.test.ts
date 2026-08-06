import { describe, it, expect } from "vitest";
import {
  getArch,
  getAllArchitectures,
  getDefaultTemplate,
  getTemplateValues,
  matchTemplate,
  serializeValue,
  buildYaml,
  formatParamCount,
  formatWeightMB,
  parseCSV,
  estimateParamsFor,
} from "../architectures";

describe("getArch", () => {
  it("returns rrdb_esrgan entry", () => {
    const def = getArch("rrdb_esrgan");
    expect(def.displayName).toBe("RRDB-ESRGAN");
    expect(def.fields.length).toBeGreaterThan(0);
  });
  it("returns swinir entry", () => {
    const def = getArch("swinir");
    expect(def.displayName).toBe("SwinIR");
  });
});

describe("getAllArchitectures", () => {
  it("returns both architectures", () => {
    expect(getAllArchitectures().length).toBe(2);
  });
});

describe("getDefaultTemplate", () => {
  it("returns the recommended template", () => {
    const def = getArch("rrdb_esrgan");
    const tpl = getDefaultTemplate(def);
    expect(tpl.recommended).toBe(true);
    expect(tpl.id).toBe("medium");
  });
});

describe("getTemplateValues", () => {
  it("returns a copy of the template values for rrdb medium", () => {
    const def = getArch("rrdb_esrgan");
    const vals = getTemplateValues(def, "medium");
    expect(vals.num_feat).toBe(64);
    expect(vals.num_block).toBe(23);
  });
  it("returns a copy of the template values for swinir medium", () => {
    const def = getArch("swinir");
    const vals = getTemplateValues(def, "medium");
    expect(vals.embed_dim).toBe(180);
    expect(vals.depths).toBe("6,6,6,6,6,6");
  });
  it("does not mutate the shared template object", () => {
    const def = getArch("swinir");
    const a = getTemplateValues(def, "medium");
    const b = getTemplateValues(def, "medium");
    a.embed_dim = 999;
    expect(b.embed_dim).toBe(180);
  });
});

describe("matchTemplate", () => {
  it("matches rrdb medium at default values", () => {
    const def = getArch("rrdb_esrgan");
    const vals = getTemplateValues(def, "medium");
    const result = matchTemplate(def, vals);
    expect(result).toBe("medium");
  });
  it("returns null after tweaking a parameter", () => {
    const def = getArch("rrdb_esrgan");
    const vals = getTemplateValues(def, "medium");
    vals.num_feat = 128;
    expect(matchTemplate(def, vals)).toBeNull();
  });
  it("ignores matchIgnoreKeys (scale)", () => {
    const def = getArch("rrdb_esrgan");
    const vals = getTemplateValues(def, "medium");
    vals.scale = 8;
    expect(matchTemplate(def, vals)).toBe("medium");
  });
  it("ignores rgb_mean", () => {
    const def = getArch("swinir");
    const vals = getTemplateValues(def, "medium");
    vals.rgb_mean = "0.5, 0.5, 0.5";
    expect(matchTemplate(def, vals)).toBe("medium");
  });
  it("matches swinir medium at default values", () => {
    const def = getArch("swinir");
    const vals = getTemplateValues(def, "medium");
    expect(matchTemplate(def, vals)).toBe("medium");
  });
});

describe("serializeValue", () => {
  it("passes non-csv scalars through", () => {
    const def = getArch("rrdb_esrgan");
    expect(serializeValue(def, "num_feat", 64)).toBe(64);
    expect(serializeValue(def, "scale", 4)).toBe(4);
  });
  it("converts csv string to number array", () => {
    const def = getArch("swinir");
    const result = serializeValue(def, "depths", "6,6,6,6,6,6");
    expect(result).toEqual([6, 6, 6, 6, 6, 6]);
  });
  it("omits empty csv when emptyCsvOmit is true", () => {
    const def = getArch("swinir");
    expect(serializeValue(def, "rgb_mean", "")).toBeUndefined();
    expect(serializeValue(def, "rgb_mean", "   ")).toBeUndefined();
  });
  it("passes null/undefined for non-csv through", () => {
    const def = getArch("rrdb_esrgan");
    expect(serializeValue(def, "num_feat", null)).toBeNull();
    expect(serializeValue(def, "num_feat", undefined)).toBeUndefined();
  });
});

describe("buildYaml", () => {
  it("emits architecture key instead of type", () => {
    const def = getArch("rrdb_esrgan");
    const yaml = buildYaml(def, { num_feat: 64, num_block: 23, num_grow_ch: 32, scale: 4, num_in_ch: 3, num_out_ch: 3 }, "my_model");
    expect(yaml).toContain("architecture: rrdb_esrgan");
    expect(yaml).not.toContain("type:");
  });
  it("includes the model name", () => {
    const def = getArch("swinir");
    const yaml = buildYaml(def, { embed_dim: 180 }, "my_swinir");
    expect(yaml).toContain("name: my_swinir");
  });
  it("renders csv fields as arrays", () => {
    const def = getArch("swinir");
    const yaml = buildYaml(def, { embed_dim: 180, depths: "6,6,6,6,6,6", scale: 4 }, "");
    expect(yaml).toContain("depths: [6, 6, 6, 6, 6, 6]");
  });
  it("omits empty csv fields", () => {
    const def = getArch("swinir");
    const yaml = buildYaml(def, { embed_dim: 180, rgb_mean: "", scale: 4 }, "");
    expect(yaml).not.toContain("rgb_mean:");
  });
});

describe("estimateParamsFor", () => {
  it("estimates rrdb_esrgan params", () => {
    const p = estimateParamsFor("rrdb_esrgan", { num_feat: 64, num_block: 23, num_grow_ch: 32 });
    expect(p).toBeCloseTo(16.7, 0);
  });
  it("estimates swinir params", () => {
    const p = estimateParamsFor("swinir", { embed_dim: 180, depths: "6,6,6,6,6,6" });
    expect(p).toBeCloseTo(11.8, 0);
  });
  it("returns 0 for unknown arch", () => {
    expect(estimateParamsFor("unknown", {})).toBe(0);
  });
});

describe("formatParamCount", () => {
  it("formats millions", () => expect(formatParamCount(16.7)).toBe("16.7 M"));
  it("formats billions", () => expect(formatParamCount(1750)).toBe("1.8 B"));
  it("formats thousands", () => expect(formatParamCount(0.5)).toBe("500 K"));
});

describe("formatWeightMB", () => {
  it("converts paramsM to fp32 MB", () => {
    // 16.7M params × 4 bytes = 66.8 MB
    expect(formatWeightMB(16.7)).toBe("66.8");
  });
});

describe("parseCSV", () => {
  it("parses comma-separated numbers", () => {
    expect(parseCSV("6,6,6,6,6,6")).toEqual([6, 6, 6, 6, 6, 6]);
  });
  it("handles whitespace", () => {
    expect(parseCSV(" 6 , 6 ")).toEqual([6, 6]);
  });
  it("returns empty array for empty string", () => {
    expect(parseCSV("")).toEqual([]);
  });
  it("filters NaN values", () => {
    const nums = parseCSV("6,abc,6");
    expect(nums).toEqual([6, 6]);
  });
});

describe("derive", () => {
  it("recomputes num_heads when embed_dim changes", () => {
    const def = getArch("swinir");
    const result = def.derive!("embed_dim", 180, { embed_dim: 180, depths: "6,6,6,6,6,6" });
    // getNumHeads(180) = 5 (180/5 = 32, 5 divides 180 evenly)
    expect(result.num_heads).toBe("5,5,5,5,5,5");
  });
  it("recomputes num_heads when depths count changes", () => {
    const def = getArch("swinir");
    const result = def.derive!("depths", "4,4,4,4", { embed_dim: 180, depths: "4,4,4,4" });
    expect(result.num_heads).toBe("5,5,5,5");
  });
  it("returns empty for unrelated keys", () => {
    const def = getArch("swinir");
    expect(def.derive!("scale", 4, {})).toEqual({});
  });
});