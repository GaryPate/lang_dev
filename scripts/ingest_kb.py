from healthbot.ingestion import ingest_documents

if __name__ == "__main__":
    count = ingest_documents("data/medical_kb")
    print(f"Ingested {count} chunks")
