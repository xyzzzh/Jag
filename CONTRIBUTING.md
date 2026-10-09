# Contributing / 参与贡献

Contributions to models, data preparation, training, and evaluation are welcome. Describe the change and its validation in your pull request, and update both README languages when public instructions change.

欢迎改进模型、数据准备、训练与评估。在合并请求中说明修改内容与验证结果；公共使用方法改变时，请同步更新中英文 README。

Run the checks with the project environment:

使用项目环境运行检查：

```bash
bash scripts/docker.sh run --rm jag python -m unittest discover -s tests -v
```

Keep datasets, weights, generated outputs, and credentials outside commits. Preserve result-block markers. Contributions use the project's [Apache 2.0 license](LICENSE).

请勿提交数据集、权重、生成产物或凭据。保留结果块标记。贡献采用项目的 [Apache 2.0 许可证](LICENSE)。
