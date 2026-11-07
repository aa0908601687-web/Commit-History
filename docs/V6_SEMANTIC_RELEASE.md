# V6 semantic-release policy

This workflow creates GitHub Releases and Git tags only; it does not publish npm packages or run lottery prediction.

Release is triggered only when the existing V6 no-cache OpenFE CUDA-Q E2E workflow succeeds on a push to main. The workflow checks that main still points to the exact E2E-verified commit before releasing. A subsequent push can be evaluated by its own E2E run.

Conventional commits determine versions: `fix:` for patch, `feat:` for minor, and `BREAKING CHANGE:` footer for major. Other commits may not produce a release.

GitHub Actions requires write permission for contents and access to create releases. Branch protection and required E2E checks should remain enabled. Review workflow_run permissions carefully; never execute untrusted pull-request artifacts or scripts in this privileged workflow.

The existing E2E suite includes checks against historical and live data; success only confirms the assertions in that workflow, not predictive accuracy, official production deployment, or a completed Rocq formal proof.
