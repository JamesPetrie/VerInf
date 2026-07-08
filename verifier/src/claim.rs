//! Claim model — deserializes the JSON that protocol.claims_to_json dumps, into
//! the public structure the compile handlers read. Explicit serde_json::Value
//! parsing (no derive macros) so the field-kind decoding is auditable.
//!
//! Schema (per field, tagged):
//!   {"var":[row_start,length]} | {"table":{...}} | {"config":{k:int}} |
//!   {"list":[field,...]} | {"none":true} | {"bool":b} | {"str":s} | {"scalar":n}

use serde_json::Value;
use std::collections::HashMap;

use crate::protocol::Config;

/// A committed variable: just the (row_start, length) the handlers need.
#[derive(Clone, Copy)]
pub struct Var {
    pub row_start: usize,
    pub length: usize,
}

/// A LogUp table's public structure. `id` (= mult_var.row_start) is the stable
/// dedup key (JSON loses Python object identity).
pub struct Table {
    pub id: usize,
    pub t_len: usize,
    /// The key domain T. None when T = range(t_len) (the common case — the
    /// verifier reconstructs T[j]=j and the array is never transmitted). Some
    /// only for a non-range fallback (unused today). See protocol._ser_table.
    pub t: Option<Vec<u64>>,
    pub t_y: Option<Vec<u64>>,
    pub mult_var: Var,
    pub w_var: Var,
    pub z_vars: Vec<Var>,
    pub alpha: u64,
    pub beta: u64,
}

impl Table {
    /// T[j] — reconstructed as j for a range table, else read from the array.
    #[inline]
    pub fn t_at(&self, j: usize) -> u64 {
        match &self.t { Some(a) => a[j], None => j as u64 }
    }
}

/// One claim: op name + its fields (kept as parsed Values, read by accessors).
pub struct Claim {
    pub op: String,
    fields: HashMap<String, Value>,
}

fn as_u64(v: &Value) -> u64 {
    // JSON numbers from Python ints up to 2^64−1; serde_json stores them as u64.
    v.as_u64().expect("expected u64 field")
}

/// A [row_start, length] pair (used for Variables inside a Table, where
/// _ser_table dumps them as bare arrays, not {"var":...}).
fn var_from_pair(a: &Value) -> Var {
    let a = a.as_array().unwrap();
    Var { row_start: a[0].as_u64().unwrap() as usize,
          length:    a[1].as_u64().unwrap() as usize }
}

fn parse_var(v: &Value) -> Var {
    var_from_pair(v.get("var").expect("expected var field"))
}

fn parse_table(v: &Value) -> Table {
    let t = v.get("table").expect("expected table field");
    let nums = |k: &str| t[k].as_array().unwrap().iter().map(as_u64).collect::<Vec<u64>>();
    let t_len = as_u64(&t["T_len"]) as usize;
    let is_range = t["T_range"].as_bool().unwrap_or(false);
    Table {
        id: as_u64(&t["id"]) as usize,
        t_len,
        t: if is_range { None } else { Some(nums("T")) },
        t_y: if t["T_Y"].is_null() { None } else { Some(nums("T_Y")) },
        mult_var: var_from_pair(&t["mult_var"]),
        w_var: var_from_pair(&t["w_var"]),
        z_vars: t["z_vars"].as_array().unwrap().iter().map(var_from_pair).collect(),
        alpha: as_u64(&t["alpha"]),
        beta: as_u64(&t["beta"]),
    }
}

impl Claim {
    /// A variable field by name (e.g. claim.var("x")).
    pub fn var(&self, name: &str) -> Var {
        parse_var(&self.fields[name])
    }
    /// An optional variable field — None when the JSON tagged it {"none":true}.
    pub fn opt_var(&self, name: &str) -> Option<Var> {
        let v = &self.fields[name];
        if v.get("none").is_some() { None } else { Some(parse_var(v)) }
    }
    /// A list-of-variables field (e.g. rmsnorm s_lo_chunks).
    pub fn var_list(&self, name: &str) -> Vec<Var> {
        self.fields[name]["list"].as_array().unwrap().iter().map(parse_var).collect()
    }
    /// A table field by name.
    pub fn table(&self, name: &str) -> Table {
        parse_table(&self.fields[name])
    }
    /// A scalar field by name.
    pub fn scalar(&self, name: &str) -> u64 {
        as_u64(&self.fields[name]["scalar"])
    }
    /// An optional scalar — None if the field is absent or tagged {"none":true}
    /// (e.g. AddClaim.public_rhs: present only for the reveal pin).
    pub fn opt_scalar(&self, name: &str) -> Option<u64> {
        match self.fields.get(name) {
            None => None,
            Some(v) if v.get("none").is_some() => None,
            Some(v) => v.get("scalar").map(as_u64),
        }
    }
    /// A bool field by name.
    pub fn boolean(&self, name: &str) -> bool {
        self.fields[name]["bool"].as_bool().unwrap()
    }
    /// A list-of-int field (e.g. token_ids), stored as {"list":[{scalar},...]}.
    pub fn int_list(&self, name: &str) -> Vec<u64> {
        self.fields[name]["list"].as_array().unwrap().iter()
            .map(|e| as_u64(&e["scalar"])).collect()
    }
    /// A config scalar (e.g. claim.cfg_int("config", "s_y")).
    pub fn cfg_int(&self, field: &str, key: &str) -> u64 {
        as_u64(&self.fields[field]["config"][key])
    }
    /// Float config field with a default for proofs predating its
    /// serialization (e.g. RoPEConfig.base, absent in older dumps).
    pub fn cfg_f64_or(&self, field: &str, key: &str, default: f64) -> f64 {
        self.fields[field]["config"][key].as_f64().unwrap_or(default)
    }
}

impl Claim {
    /// Every table field of this claim (any field tagged {"table":...}).
    pub fn tables(&self) -> Vec<Table> {
        self.fields.values()
            .filter(|v| v.get("table").is_some())
            .map(|v| parse_table(v))
            .collect()
    }

    /// Every Variable reachable from this claim — direct {"var"}, list-of-var,
    /// and a table's mult_var/w_var/z_vars. Mirrors protocol._walk_vars (for
    /// m_total). Order-independent (used only for a max).
    pub fn all_vars(&self) -> Vec<Var> {
        let mut out = Vec::new();
        for v in self.fields.values() {
            if v.get("var").is_some() {
                out.push(parse_var(v));
            } else if let Some(items) = v.get("list").and_then(|x| x.as_array()) {
                for it in items {
                    if it.get("var").is_some() { out.push(parse_var(it)); }
                }
            } else if v.get("table").is_some() {
                let t = parse_table(v);
                out.push(t.mult_var); out.push(t.w_var);
                out.extend(t.z_vars);
            }
        }
        out
    }

    /// Override each table field's α/β with the s_op-derived values (keyed by
    /// table id = mult_var.row_start), since the JSON's α/β are stale.
    pub fn patch_table_ab(&mut self, ab: &HashMap<usize, (u64, u64)>) {
        for v in self.fields.values_mut() {
            if v.get("table").is_some() {
                let id = v["table"]["mult_var"][0].as_u64().unwrap() as usize;
                if let Some(&(alpha, beta)) = ab.get(&id) {
                    v["table"]["alpha"] = Value::from(alpha);
                    v["table"]["beta"] = Value::from(beta);
                }
            }
        }
    }
}

/// The full deserialized input: cfg + the OPERATION claim list + the explicit
/// table settle order (table ids, = Python's _distinct_tables).
pub struct ClaimSet {
    pub cfg: Config,
    pub claims: Vec<Claim>,
    pub table_order: Vec<usize>,
}

impl ClaimSet {
    /// Map id → Table, gathered from every claim's table fields (the same table
    /// id appears in multiple claims; any copy has identical data).
    pub fn tables_by_id(&self) -> HashMap<usize, Table> {
        let mut m = HashMap::new();
        for cl in &self.claims {
            for t in cl.tables() {
                m.entry(t.id).or_insert(t);
            }
        }
        m
    }
}

pub fn parse_claim_set(json: &str) -> ClaimSet {
    parse_claim_set_value(serde_json::from_str(json).expect("bad claim JSON"))
}

/// Parse from an already-deserialized Value — avoids re-serializing + re-parsing
/// a giant claims object (the 2^k LogUp tables make that round-trip cost GBs).
pub fn parse_claim_set_value(mut v: Value) -> ClaimSet {
    // Takes the Value BY VALUE and MOVES each claim's fields out (take() +
    // Map::into_iter) rather than cloning. At full-model scale the claims
    // sub-object is ~4 GB of JSON (long op-history variable names × 11k
    // claims × 128-var combine lists); cloning every field doubled the
    // resident claims data and OOM'd at >113 GB. Moving keeps one copy.
    let c = &v["cfg"];
    let cfg = Config {
        ell: as_u64(&c["ELL"]),
        k_deg: as_u64(&c["K_DEG"]),
        n_lig: as_u64(&c["N_LIG"]),
        t_queries: as_u64(&c["T_QUERIES"]) as usize,
    };
    let table_order = v["table_order"].as_array().unwrap().iter()
        .map(|x| x.as_u64().unwrap() as usize).collect();
    let claims_arr = match v["claims"].take() {
        Value::Array(a) => a,
        _ => panic!("claims must be an array"),
    };
    let claims = claims_arr.into_iter().map(|mut cl| {
        let op = cl["op"].as_str().unwrap().to_string();
        let fields = match cl["fields"].take() {
            Value::Object(m) => m.into_iter().collect::<HashMap<String, Value>>(),
            _ => HashMap::new(),
        };
        Claim { op, fields }
    }).collect();
    ClaimSet { cfg, claims, table_order }
}
