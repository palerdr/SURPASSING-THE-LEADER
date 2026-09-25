import assert from "node:assert/strict";
import test from "node:test";

import { escapeHtml } from "../src/render/escape.ts";

test("escapeHtml neutralizes every HTML-significant character", () => {
  assert.equal(
    escapeHtml(`<script data-name="'">&</script>`),
    "&lt;script data-name=&quot;&#39;&quot;&gt;&amp;&lt;/script&gt;",
  );
});

test("escapeHtml stringifies non-string display values", () => {
  assert.equal(escapeHtml(null), "null");
  assert.equal(escapeHtml(61), "61");
});
