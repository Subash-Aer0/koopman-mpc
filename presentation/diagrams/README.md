# Block diagrams (TikZ)

`kres_loop.tex` (paper's LQR + K-Res-MPC loop) and `kfull_loop.tex` (K-Full-MPC loop)
share `common.tex`, whose colours match `report/koopman_mpc.tex`.
The slides include `figures/*.svg`; the `.pdf` copies can go straight into the report.

Rebuild (from `presentation/diagrams`):

```bash
conda activate latex_tools
for f in kres_loop kfull_loop; do
  tectonic --outdir ../figures $f.tex && pdftocairo -svg ../figures/$f.pdf ../figures/$f.svg
done
```

`tectonic` downloads the Lato font package on first use. The system `pdflatex` works too
(`pdflatex -output-directory=../figures kfull_loop.tex`); without Lato installed it
falls back to the default sans font, so the text looks slightly different from the slides.
Run either from this folder, because `\input{common}` is a relative path.
