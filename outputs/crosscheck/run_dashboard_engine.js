// Runs the dashboard's own OLS engine on h1a_sample.csv (DEVIATIONS.md, 2026-10-20 cross-check).
// Design matrix as the dashboard builds it: intercept column first, then the regressors.
const fs = require("fs");
const path = require("path");
const { ols } = require("./dashboard_ols_engine.js");

const lines = fs.readFileSync(path.join(__dirname, "h1a_sample.csv"), "utf8").trim().split(/\r?\n/);
const header = lines[0].split(",");
const yName = "r_btc_noon";
const xNames = header.filter((h) => h !== "date" && h !== yName);
const rows = lines.slice(1).map((l) => l.split(","));
const y = rows.map((r) => parseFloat(r[header.indexOf(yName)]));
const X = rows.map((r) => [1, ...xNames.map((c) => parseFloat(r[header.indexOf(c)]))]);

const out = ["term,coef,se_classical,se_hc1"];
const classical = ols(X, y, "classical", true);
const robust = ols(X, y, "robust", true);
["const", ...xNames].forEach((t, i) =>
  out.push([t, classical.beta[i], classical.se[i], robust.se[i]].map(String).join(",")));
fs.writeFileSync(path.join(__dirname, "h1a_dashboard_engine.csv"), out.join("\n") + "\n");
console.log(`n = ${classical.n}, k = ${classical.k}`);
console.log(out.join("\n"));
