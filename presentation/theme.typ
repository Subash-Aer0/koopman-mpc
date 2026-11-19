// Academic slide theme for Typst (self-contained: no packages, works offline).
//
// Usage:
//   #import "theme.typ": *
//   #show: academic-theme.with(title: [...], author: [...], ...)
//   #title-slide()
//   #section-slide[Section name]
//   #slide(title: [Slide title])[ body ]
//
// Layout: 16:9, a thin progress bar and the current section at the top,
// author | short title | page / total at the bottom.

// Palettes. Pick one at compile time without touching the deck:
//   typst compile slides.typ                              (navy, default)
//   typst compile --input theme=iitm slides.typ slides_iitm.pdf
//   typst compile --input theme=breeze slides.typ slides_breeze.pdf
// good / warn / amber / blue are the Okabe-Ito plot colours from make_figures.py.
// good, amber, blue, vermillion and purple are legend swatches for the figures, so no palette may change
// them; warn is only used for text and cards, so a palette may override it.
#let _plot-colors = (
  good: rgb("#009E73"),
  warn: rgb("#D55E00"),
  amber: rgb("#E69F00"),
  blue: rgb("#0072B2"),
  vermillion: rgb("#D55E00"), // LQR; separate from warn, which a palette may override
  purple: rgb("#CC79A7"), // LQR + K-Res-MPC
)

#let palettes = (
  navy: (
    primary: rgb("#13294b"), // navy: titles, title slide
    accent: rgb("#0072B2"), // blue: rules, highlights
    on-primary: rgb("#0072B2"), // decorations drawn on the primary fill
    text: rgb("#1f2933"),
    muted: rgb("#6b7280"),
    light: rgb("#eef3f9"), // box backgrounds
  )
    + _plot-colors,
  // Colours sampled from the IIT Madras logo (maroon ring, ochre field).
  // The logo ochre #D4A64F is only 2.2:1 on white, so text/rules use a darker
  // ochre (4.1:1); the logo ochre is kept for decorations on maroon (4.6:1).
  iitm: (
    primary: rgb("#7A1F1A"), // logo maroon
    accent: rgb("#A8741F"), // darkened logo ochre
    on-primary: rgb("#D4A64F"), // logo ochre
    text: rgb("#2B1D1A"), // warm near-black (logo outline)
    muted: rgb("#7A6A62"),
    light: rgb("#F7EFE2"), // cream
  )
    + _plot-colors
    + (
      warn: rgb("#C0281D"), // logo lotus red; Okabe-Ito orange is too close to ochre
    ),
  // Light, airy look: off-white paper with a faint grid, navy ink, teal accent.
  // Teal is darkened from #1A9E8F (3.1:1 on the paper) to #0E7C70 (4.8:1).
  breeze: (
    primary: rgb("#111B2E"), // deep navy ink
    accent: rgb("#0E7C70"), // teal
    on-primary: rgb("#0E7C70"),
    text: rgb("#1F2937"),
    muted: rgb("#5B6472"),
    light: rgb("#E6F2EF"), // mint tint for boxes
    page: rgb("#F8F8F6"), // off-white paper
    grid: rgb("#ECEDE9"), // faint grid lines
    title-fill: rgb("#F8F8F6"), // light title slide instead of a solid block
    title-text: rgb("#111B2E"),
    title-muted: rgb("#5B6472"),
    section-fill: rgb("#F8F8F6"),
  )
    + _plot-colors,
)

// Layout keys a palette may omit; the defaults reproduce the original navy look
// (white pages, solid primary title slide, tinted section slides).
#let palette = {
  let p = palettes.at(sys.inputs.at("theme", default: "iitm"))
  (
    (
      page: white,
      grid: none,
      title-fill: p.primary,
      title-text: white,
      title-muted: white.darken(20%),
      section-fill: p.light,
    )
      + p
  )
}

// Faint square grid drawn behind every page when the palette sets `grid`.
#let _grid-cell = 0.9cm
#let _background = if palette.grid != none {
  rect(width: 100%, height: 100%, stroke: none, fill: tiling(size: (_grid-cell, _grid-cell), place(rect(
    width: _grid-cell,
    height: _grid-cell,
    stroke: (top: 0.5pt + palette.grid, left: 0.5pt + palette.grid),
  ))))
}

#let _section = state("academic-section", none)
#let _meta = state("academic-meta", (:))

// ---------------------------------------------------------------- page furniture
#let _header = context {
  let total = counter(page).final().first()
  let current = counter(page).get().first()
  place(top + left, dx: -1.6cm, rect(width: 100% + 3.2cm, height: 4pt, fill: palette.light))
  place(top + left, dx: -1.6cm, rect(
    width: (100% + 3.2cm) * (current / calc.max(total, 1)),
    height: 4pt,
    fill: palette.accent,
  ))
  let sec = _section.get()
  if sec != none {
    v(0.55cm)
    align(right, text(size: 11pt, fill: palette.muted, smallcaps(sec)))
  }
}

#let _footer = context {
  let meta = _meta.get()
  set text(size: 10pt, fill: palette.muted)
  line(length: 100%, stroke: 0.5pt + palette.light)
  v(-0.35em)
  grid(
    columns: (1fr, 2fr, 1fr),
    align(left, meta.at("author", default: [])),
    align(center, meta.at("short-title", default: [])),
    align(right, [#counter(page).display() / #counter(page).final().first()]),
  )
}

// ---------------------------------------------------------------- main show rule
#let academic-theme(
  title: [],
  subtitle: none,
  author: [],
  institution: none,
  date: none,
  short-title: none,
  body,
) = {
  set document(title: if type(title) == str { title } else { "Slides" })
  set page(
    paper: "presentation-16-9",
    fill: palette.page,
    background: _background,
    margin: (x: 1.6cm, top: 1.5cm, bottom: 1.3cm),
    header: _header,
    header-ascent: 0.2cm,
    footer: _footer,
    footer-descent: 0.4cm,
  )
  set text(font: ("Lato", "Noto Sans", "DejaVu Sans"), size: 16pt, fill: palette.text)
  show math.equation: set text(font: ("New Computer Modern Math", "DejaVu Math TeX Gyre"))
  set par(leading: 0.62em, justify: false)
  set list(
    marker: (text(fill: palette.accent)[●], text(fill: palette.accent)[‣]),
    indent: 0.2em,
    body-indent: 0.5em,
    spacing: 0.75em,
  )
  set enum(numbering: n => text(fill: palette.accent, weight: "bold")[#n.], spacing: 0.75em)
  show strong: set text(fill: palette.primary)
  set table(stroke: none, inset: (x: 0.55em, y: 0.38em))
  show table.cell.where(y: 0): set text(weight: "bold", fill: palette.primary)
  show raw: set text(font: ("Ubuntu Mono", "DejaVu Sans Mono"), size: 0.95em)
  show link: set text(fill: palette.accent)

  _meta.update((
    title: title,
    subtitle: subtitle,
    author: author,
    institution: institution,
    date: date,
    short-title: if short-title == none { title } else { short-title },
  ))
  body
}

// ---------------------------------------------------------------- slide types
#let title-slide() = context {
  let m = _meta.get()
  page(fill: palette.title-fill, header: none, footer: none, margin: 2.2cm)[
    #set text(fill: palette.title-text)
    #place(top + left, dx: -2.2cm, dy: -2.2cm, rect(width: 100% + 4.4cm, height: 0.35cm, fill: palette.on-primary))
    #v(1fr)
    #text(size: 34pt, weight: "bold", m.title)
    #if m.subtitle != none [
      #v(0.2em)
      #text(size: 20pt, fill: palette.title-muted, m.subtitle)
    ]
    #v(0.8em)
    #line(length: 30%, stroke: 2pt + palette.on-primary)
    #v(0.8em)
    #text(size: 18pt, weight: "bold", m.author) \
    #if m.institution != none { text(size: 15pt, fill: palette.title-muted, m.institution) } \
    #if m.date != none { text(size: 14pt, fill: palette.title-muted, m.date) }
    #v(1.4fr)
  ]
}

#let section-slide(name) = {
  _section.update(name)
  page(fill: palette.section-fill, header: none, footer: none)[
    #v(1fr)
    #context {
      let n = counter("academic-section-n")
      n.step()
      text(size: 16pt, fill: palette.accent, weight: "bold")[Part #(n.get().first() + 1)]
    }
    #v(0.1em)
    #text(size: 32pt, weight: "bold", fill: palette.primary, name)
    #v(0.2em)
    #line(length: 25%, stroke: 2.5pt + palette.accent)
    #v(1fr)
  ]
}

#let slide(title: none, body) = {
  pagebreak(weak: true)
  if title != none {
    block(width: 100%, below: 0.7em)[
      #text(size: 23pt, weight: "bold", fill: palette.primary, title)
      #v(-0.45em)
      #line(length: 100%, stroke: 1.5pt + palette.accent)
    ]
  }
  body
}

// ---------------------------------------------------------------- content helpers
// Shaded box with an optional heading, for definitions, claims and take-aways.
#let card(title: none, color: palette.accent, body) = block(
  width: 100%,
  fill: palette.light,
  inset: (x: 0.8em, y: 0.65em),
  radius: 4pt,
  stroke: (left: 3.5pt + color),
)[
  #if title != none {
    text(weight: "bold", fill: color, title)
    v(-0.3em)
  }
  #body
]

// One-line take-away at the bottom of a slide.
// Placed at the bottom of the content area, so it never pushes content to a new page;
// leave room for it (about 2.2 cm) when sizing figures.
#let takeaway(body) = place(bottom + left, block(width: 100%, fill: palette.primary, inset: 0.55em, radius: 4pt, {
  set text(fill: white, size: 15pt)
  show strong: set text(fill: white)
  [*Take-away.* #body]
}))

#let muted(body) = text(fill: palette.muted, size: 0.85em, body)
#let hl(body) = text(fill: palette.accent, weight: "bold", body)
#let bad(body) = text(fill: palette.warn, weight: "bold", body)
#let good(body) = text(fill: palette.good, weight: "bold", body)

// Coloured square used as a legend key in text and tables.
#let swatch(c) = box(rect(width: 0.75em, height: 0.75em, fill: c, radius: 1.5pt), baseline: 0.1em)

// Signal-flow block for diagrams.
#let blk(body, fill: palette.light, stroke: palette.accent) = box(
  fill: fill,
  stroke: 1.2pt + stroke,
  inset: (x: 0.5em, y: 0.45em),
  radius: 4pt,
  align(center, text(size: 13pt, body)),
)
#let arr(label: none) = box(baseline: -0.2em)[
  #set text(size: 11pt, fill: palette.muted)
  #stack(dir: ttb, spacing: 0.1em, align(center, if label != none { label } else { [] }), text(
    size: 18pt,
    fill: palette.accent,
  )[#sym.arrow.r.long])
]
