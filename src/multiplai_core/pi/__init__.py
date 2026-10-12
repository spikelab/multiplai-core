"""Pi (``pi --mode rpc``) as an agent backend for :func:`multiplai_core.agent_runner.run_agent`.

``run_agent(..., model="pi:<model-id>@<provider>")`` imports
:mod:`multiplai_core.pi.backend` on first use, which registers the ``pi``
backend. Nothing here is imported by ``import multiplai_core`` or by
``import multiplai_core.agent_runner``.

Installing and configuring Pi (the binary, ``~/.pi/agent/models.json``,
provider keys) is the user's job and is not covered here.
"""
