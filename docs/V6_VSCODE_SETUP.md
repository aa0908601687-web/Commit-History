# V6 OF3Q VS Code workspace

This is an editor configuration, not an installation of VS Code or model dependencies. Works with the existing `main` branch, independently of the pending non-variational CUDA-Q removal PR #65.

1. Install VS Code and Python 3.11+ on your computer and open the repository folder.
2. Use the recommended extensions and choose an interpreter.
3. Run **Terminal > Run Task > V6: Exact-search unit tests** to test the exhaustive 8-choose-6 optimizer.
4. Use **Run and Debug > V6: Debug 8-to-6 exhaustive search tests** to step through unit tests.
5. The full V6 regression task may require additional external AI dependencies; success or failure should be checked in GitHub Actions as well.

The V6 selection modernization is tracked separately in PR #65. Only that PR can remove non-variational CUDA-Q from the executable application. Workspace settings alone neither deploy the new algorithm nor improve fair lottery winning odds.
