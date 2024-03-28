# Rubin Observatory SIT-Com AOS Analysis Repository

This repository is to store and organize all the data analysis scripts, notebooks and any associated methods which are related to an AOS On-Sky Commissioning Test with a JIRA ticket. The reason for creation of this repo is to make it easier to organize and execute each data analysis process in a systematic fashion.

To keep the size of the repository small and therefore faster to clone/manage, it is recommended to clear the notebooks of rendered content before committing via git. It is also recommended that notebooks and/or associated methods that may need to be used by others contain at least a minimal amount of documentation and/or comments to provide sufficient context and instructions.


## Notebooks

User notebooks should be stored in the notebooks directory.
Please, remember to stop the Kernel and clean all the notebooks outputs before committing and pushing them to GitHub.
The folder structure inside the `notebooks` directory should approximately reflect the organization in the [AOS On-sky Test Plan]. Each block of tests already has a different folder.
Please, remember to reference the technote each notebook contributes to at the beginning of the notebook. 

Each notebook's name should start with the test name, a dash, and short name that represent the notebook purpose.
This will help quick assesment of each notebook's function while keeping the information required to find the actual Test Case.
Do not use spaces in the filename.
Instead, replace spaces with underlines (`_`).


## Methods

User methods developed to support notebooks should be stored in the python directory.
It is strongly recommended to follow Rubin development formats/practices to standardize behavior and minimize the overhead when sharing/running each others code.
This repo is eups compatible.
If a user wishes to develop their own support methods, this repo must be setup prior to importing them.

One way to setup this repo is to add the following to the `~/notebooks/.user_setups` file:

    setup -j notebooks_vandv -r ~/notebooks/lsst-sitcom/aos_analysis

You can replace `~/notebooks/lsst-sitcom/aos_analysis` with the directory where this file is located.


## Developer Guide

We will try to adopt most of the practices/workflow adopted by the Telecope and Site and Data-Management teams.
Here are the links for both of them:

- https://tssw-developer.lsst.io/
- https://developer.lsst.io/

Here are a few quick points to keep an eye on:

1. We want to follow [TS Development Workflow] (JIRA Ticket, new branch `ticket/PROJ-????`, Pull Request).
2. When writing documentation or text in notebooks, try to use [Semantic Like Breaks] for clarity.
3. For code standards, let's use the [PEP-8] as a reference.
4. When writing new plots/tools, consider the [Rule of Three] to avoid duplication/repetition.
5. When writing commit messages, consider these [Best Commit Practices].


## Attribution

This repository has been inspired by [notebooks_vandv]



[notebooks_vandv]: https://github.com/lsst-sitcom/notebooks_vandv
[TS Development Workflow]: https://tssw-developer.lsst.io/work_management/development_workflow.html#development-workflow
[Semantic Like Breaks]: https://sembr.org/
[PEP-8]: https://peps.python.org/pep-0008/
[Rule of Three]: https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming)
[Best Commit Practices]: https://www.linkedin.com/pulse/7-best-practices-writing-good-git-commitmessages-kirinyet-brian
