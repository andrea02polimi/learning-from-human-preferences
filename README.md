# Learning from Human Preferences — Modernized Fork

This repository is a **fork of the original implementation** of *Deep Reinforcement Learning from Human Preferences*.

Original repository:  
https://github.com/mrahtz/learning-from-human-preferences

The original code reproduces the method described in the paper:

> **Deep Reinforcement Learning from Human Preferences**  
> Christiano et al., 2017  
> https://arxiv.org/abs/1706.03741

---

# Purpose of this Fork

The goal of this fork is to **modernize and maintain the original implementation**, which was written for an outdated software stack.

The original repository depends on:

- TensorFlow 1
- an old Gym API
- deprecated logging utilities
- legacy environment wrappers

These dependencies make the project difficult to run on modern systems.

This fork aims to update the project while preserving the original algorithmic structure.

---

# Goals

The main objectives of this fork are:

- update the codebase to **modern Python environments**
- replace **deprecated dependencies**
- improve **project structure and modularity**
- make the repository easier to **reuse as a library**
- ensure **reproducibility of the original experiments**

Planned improvements include:

- migration away from TensorFlow 1
- compatibility with modern RL environments
- improved logging and visualization
- cleaner packaging and installation

---

# Current Status

The repository is currently being refactored to:

- reorganize the codebase into a proper Python package
- clean up the project structure
- fix compatibility issues with modern dependencies

Some components of the original implementation are still under migration.

---

# Original Authors

All credit for the original implementation goes to the authors of the repository:

https://github.com/mrahtz/learning-from-human-preferences

This fork is intended purely as a **maintenance and modernization effort**.

---

# License

This repository preserves the **same license as the original project**.

See the `LICENSE` file for details.