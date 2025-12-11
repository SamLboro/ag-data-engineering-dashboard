# Agricultural Commodities Real-Time Data Platform

A production-grade data engineering pipeline that ingests live commodity prices (corn, wheat, soy futures) and weather data, processes them using Apache Spark and Kafka streaming, and delivers analytics-ready datasets to Snowflake—mirroring the infrastructure used by top agricultural trading desks at firms like Cargill, Vitol, and Trafigura. Built with Python, PySpark, Kafka, Airflow, AWS (S3/EKS), Docker, Kubernetes, and Terraform to demonstrate end-to-end data platform design from API ingestion through cloud warehousing.

**Tech Stack:** Python | PySpark | Kafka | Airflow/Prefect | Snowflake | AWS (S3, EKS, IAM) | Docker | Kubernetes | Terraform | Streamlit

## Project Overview

This project replicates the real-time data infrastructure used by agricultural commodity trading desks to monitor market prices and weather signals that drive crop yields. The pipeline continuously ingests CME futures data (corn/wheat/soy) and gridded weather datasets (temperature, precipitation, soil moisture from NOAA/ECMWF), streams them through Kafka topics, processes and enriches the data using Apache Spark (both batch and structured streaming), orchestrates workflows with Airflow/Prefect, and loads the results into Snowflake for analytics. The system calculates features like Growing Degree Days (GDD), precipitation anomalies, and weather-stress indicators that correlate with price movements—exactly the signals quantitative ag traders use to make million-dollar decisions. Deployed on AWS using Docker and Kubernetes with infrastructure-as-code (Terraform), this platform demonstrates production-grade data engineering practices including idempotent pipelines, schema validation, monitoring, and cost optimization strategies used at firms managing billions in agricultural commodities exposure.

## Architecture Diagram
[Coming soon - will show data flow from API → Kafka → Spark → Snowflake]

## Current Status
- [x] Phase 0: Project scoped and repo initialized
- [ ] Phase 1: Local data exploration (Pandas)
- [ ] Phase 2: Spark transformations (local)
- [ ] Phase 3: Kafka streaming (Docker Compose)
- [ ] Phase 4: Workflow orchestration (Prefect)
- [ ] Phase 5: Cloud deployment (AWS + Snowflake)

## Quick Start
[Instructions for running locally - tbd]

## Sample Output
[Screenshots of data/dashboards - adding these once built]

## Lessons Learned
[Documenting challenges and solutions - updated weekly in LEARNINGS.md]
