-- Pandoc Lua filter (docx build only): rewrite display-math \begin{array}
-- blocks as real tables with inline-math cells.
--
-- Why: pandoc encodes TeX arrays as OMML matrices, and Google Docs' equation
-- importer has no matrix support, so the Appendix B claim listings vanished
-- on upload. Word tables with non-matrix inline math import cleanly.
-- \hline rows are dropped (OMML has none either); the visual grouping is
-- carried by the label rows. PDF and GitHub builds do not use this filter.

local function split_top_level(s, sep)
  -- split on sep at brace depth 0, ignoring escaped separators (\&)
  local parts, depth, cur, i = {}, 0, "", 1
  while i <= #s do
    local c = s:sub(i, i)
    if c == "\\" and i < #s then
      cur = cur .. s:sub(i, i + 1); i = i + 2
    else
      if c == "{" then depth = depth + 1
      elseif c == "}" then depth = depth - 1 end
      if c == sep and depth == 0 then
        table.insert(parts, cur); cur = ""
      else
        cur = cur .. c
      end
      i = i + 1
    end
  end
  table.insert(parts, cur)
  return parts
end

local function array_to_table(colspec, body)
  -- column alignments from e.g. "lll|ccc" (pipes drop; they were the rules)
  local aligns = {}
  for ch in colspec:gmatch("[lcr]") do
    aligns[#aligns + 1] = (ch == "c" and pandoc.AlignCenter)
      or (ch == "r" and pandoc.AlignRight) or pandoc.AlignLeft
  end
  local rows = {}
  for _, rowtex in ipairs(split_top_level(body, "\n")) do
    rowtex = rowtex:gsub("\\\\%s*$", ""):gsub("^%s+", ""):gsub("%s+$", "")
    if rowtex ~= "" and rowtex ~= "\\hline" then
      local cells = {}
      for _, celltex in ipairs(split_top_level(rowtex, "&")) do
        celltex = celltex:gsub("^%s+", ""):gsub("%s+$", "")
        if celltex == "" then
          cells[#cells + 1] = {}
        else
          cells[#cells + 1] = { pandoc.Plain { pandoc.Math("InlineMath", celltex) } }
        end
      end
      while #cells < #aligns do cells[#cells + 1] = {} end
      rows[#rows + 1] = cells
    end
  end
  if #rows == 0 then return nil end
  local tbl = pandoc.utils.from_simple_table(
    pandoc.SimpleTable({}, aligns, {}, {}, rows))
  -- from_simple_table derives colspecs from the header row; with no header
  -- it returns an empty colspec list and the docx writer drops every column
  local colspecs = {}
  for i = 1, #aligns do colspecs[i] = { aligns[i], pandoc.ColWidthDefault } end
  tbl.colspecs = colspecs
  return tbl
end

local function aligned_to_paras(body)
  -- \begin{aligned} also becomes an OMML matrix; split its rows into
  -- separate one-line display equations instead (alignment tabs dropped)
  local paras = {}
  for _, rowtex in ipairs(split_top_level(body, "\n")) do
    rowtex = rowtex:gsub("\\\\%s*$", ""):gsub("^%s+", ""):gsub("%s+$", "")
    if rowtex ~= "" then
      local line = rowtex:gsub("&", " ")
      paras[#paras + 1] = pandoc.Para { pandoc.Math("DisplayMath", line) }
    end
  end
  if #paras == 0 then return nil end
  return paras
end

function Para(el)
  if #el.content ~= 1 then return nil end
  local m = el.content[1]
  if m.t ~= "Math" or m.mathtype ~= "DisplayMath" then return nil end
  local colspec, body = m.text:match("\\begin{array}{([^}]*)}(.*)\\end{array}")
  if colspec then
    body = body:gsub("^%s*\n", "")
    local ok, tbl = pcall(array_to_table, colspec, body)
    if ok and tbl then return tbl end
    return nil
  end
  local abody = m.text:match("\\begin{aligned}(.*)\\end{aligned}")
  if abody then
    local ok, paras = pcall(aligned_to_paras, abody)
    if ok and paras then return paras end
  end
  return nil
end
