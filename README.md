# The Quranic

This AI-enabled app, helps Quran readers or people interested in the Quran, search through its verses efficiently by putting in a keyword or short description of a theme.

## Key Features
- Search Quran verses by keyword.
- Search Quran verses by description (semantic search).
- Complete display of Quran verses with English translation.

## Tools/Libraries Used
* quran.json - Obtained from npm, then imported to project folder. Acts as the data source for the English translation of the Quran.
* Sentence transformers - Helps create vector embeddings for each Quran verse. The paraphrase-multilingual-MiniLM-L12-v2 model is used to execute the task.
* Transformers.js - Helps create vector embeddings for user queries. The model, all-MiniLM-L6-v2 is imported via CDN.

## Notes
create_embeddings.py is used to create embeddings for each Quran verse. If you intend on running the project as it is, you may ignore the file.