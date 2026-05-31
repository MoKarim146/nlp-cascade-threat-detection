# Project Background

This repository is an independent AI engineering and NLP portfolio project. It explores practical cost-aware inference for linguistic threat and hate-speech detection.

The motivation is deployment-oriented: stronger transformer classifiers can improve quality, but running them on every input may be unnecessarily expensive. A cascade can route high-confidence examples through a lightweight CPU model and reserve transformer inference for uncertain cases.

The project compares classical machine learning and transformer inference as practical system-design choices. The public version focuses on reproducible API-free workflows and saved result summaries.

## Core Questions

- How far can a TF-IDF plus Logistic Regression first tier go as a cheap baseline?
- How much quality is gained by escalating uncertain cases to RoBERTa?
- How many transformer calls can be avoided while tracking Macro F1 and threat false-negative rate?
- Which results are reproducible locally without paid APIs?

## Public Scope

The public repository is designed for review as an AI/NLP engineering project. It emphasizes clean documentation, safe release hygiene, local reproducibility, and honest reporting of saved metrics.
