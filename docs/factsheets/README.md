To render factsheets in HTML,

1. Install Quarto
2. Install the glossary extension with `quarto install extension debruine/quarto-glossary` in each folder (en, fr)
3. Run `quarto preview` for each folder separately, Quarto does not support multilingual sites.


To serve them in the Panel app, what I've been able to do so far is hackish...

1. Create `assets` directory in `src/peach/frontend/`
2. Within `assets`, create symlinks to the HTML build directories of Quarto
   ```bash
   fr -> docs/factsheets/fr/_book/
   en -> docs/factsheets/en/_book/
   ```
3. Launch the panel app with support for static directories (see https://panel.holoviz.org/how_to/server/static_files.html)
   ```python
   app.show(static_dirs={"docs/fr": "./assets/fr", "docs/en": "./assets/en"})
   ```
4. In the application header, add HTML links to `index.html`
    ```html
   <a href="docs/en/index.html">Documentation (en)</a>
   ```

I'm not able to spin a docker container since it requires MinIO credentials, so I haven't been able to fold that logic into the container. I suspect that the docker logic needs to
1. install quarto and generate the HTML;
2. include the static dirs in `run.sh`. 
