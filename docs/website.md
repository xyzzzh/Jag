# Project page

The source for [xyzzzh.github.io/Jag](https://xyzzzh.github.io/Jag/) is in [site](site/). It is a static HTML, CSS, and JavaScript site.

Preview locally:

```bash
python3 -m http.server 8080 --directory docs/site
```

After committing page changes, publish with:

```bash
bash scripts/publish_site.sh
```

GitHub Pages serves the root of the `gh-pages` branch. Keep this branch generated from `docs/site` so updates retain the deployment history. Results come from `evaluation/results.json`; keep the page, README tables, and model card synchronized when measurements change.

项目页源码位于 `docs/site`，提交修改后执行上述发布命令。GitHub Pages 从 `gh-pages` 分支根目录提供页面。更新结果时，同步修改网页、中英文 README 和模型卡。
