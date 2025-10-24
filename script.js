// Import Transformers.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'; //only takes pipiline and env from the cdn

// Configure Transformers.js to use local models
env.allowLocalModels = false;
env.allowRemoteModels = true;

// State management
let quranData = null;
let allVersesWithEmbeddings = [];
let embedder = null;
let queryEmbeddingCache = {};
let currentView = 'home';
let modelLoaded = false;

// Initialize app
async function init() {
    await loadQuranData(); // Functions that need to carry out once web accessed
    await initializeModel();
}

// Load Quran data with embeddings
async function loadQuranData() {
    try {
        const response = await fetch('quran_with_embeddings.json');
        quranData = await response.json();
        
        // Flatten all verses for quick access
        allVersesWithEmbeddings = []; // Nested loop
        quranData.surahs.forEach((surah, surahIndex) => {
            surah.verses.forEach(verse => {
                allVersesWithEmbeddings.push({
                    surah_id: surahIndex + 1,
                    surah_name: surah.name,
                    surah_english: surah.transliteration,
                    verse_id: verse.id,
                    text_arabic: verse.text,
                    text_translation: verse.translation,
                    embedding: verse.embedding
                });
            });
        });
        
        console.log(`✓ Loaded ${allVersesWithEmbeddings.length} verses with embeddings`);
        displaySurahList();
    } catch (error) {
        showError('Failed to load Quran data. Make sure quran_with_embeddings.json is in the same folder.');
    }
}

// Initialize the local AI model
async function initializeModel() {
    if (modelLoaded) return;
    
    console.log('Initializing AI model');
    const modelLoading = document.getElementById('modelLoading');
    const loadingStatus = document.getElementById('loadingStatus');
    const progressFill = document.getElementById('progressFill');
    
    try {
        // Show loading UI
        modelLoading.style.display = 'block';
        
        // Create feature extraction pipeline with progress tracking
        loadingStatus.textContent = 'Downloading model (one-time, ~90MB)...';
        progressFill.style.width = '10%';
        
        embedder = await pipeline(
            'feature-extraction',
            'Xenova/all-MiniLM-L6-v2',
            {
                progress_callback: (progress) => {
                    if (progress.status === 'downloading') {
                        const percent = Math.round((progress.loaded / progress.total) * 100);
                        progressFill.style.width = percent + '%';
                        loadingStatus.textContent = `Downloading model... ${percent}%`;
                    } else if (progress.status === 'loading') {
                        progressFill.style.width = '95%';
                        loadingStatus.textContent = 'Loading model into memory...';
                    }
                }
            }
        );
        
        progressFill.style.width = '100%';
        loadingStatus.textContent = '✓ Model loaded! AI search is ready.';
        
        modelLoaded = true;
        console.log('✓ AI model loaded successfully!');
        
        // Hide loading UI after a moment
        setTimeout(() => {
            modelLoading.style.display = 'none';
        }, 1500);
        
    } catch (error) {
        console.error('Error loading model:', error);
        loadingStatus.textContent = '✗ Failed to load model. Semantic search will not be available.';
        document.getElementById('semanticRadio').disabled = true;
    }
}

// Get embedding for query using local model
async function getQueryEmbedding(query) {
    // Check cache first
    if (queryEmbeddingCache[query]) {
        console.log('Using cached embedding for:', query);
        return queryEmbeddingCache[query];
    }

    if (!embedder) {
        throw new Error('AI model not loaded yet. Please wait...');
    }

    console.log('Generating embedding for query:', query);

    try {
        // Generate embedding
        const output = await embedder(query, { pooling: 'mean', normalize: true });
        
        // Convert to regular array
        const embedding = Array.from(output.data);
        
        console.log('✓ Generated embedding, dimension:', embedding.length);
        queryEmbeddingCache[query] = embedding;
        return embedding;
        
    } catch (error) {
        console.error('Error generating query embedding:', error);
        throw error;
    }
}

// Calculate cosine similarity
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Semantic search using pre-computed embeddings (FAST!)
async function semanticSearch(query, topK = 20) {
    const content = document.getElementById('content');
    content.innerHTML = '<div class="loading">AI is analyzing your query</div>';

    try {
        // Get embedding for the query
        const queryEmbedding = await getQueryEmbedding(query);

        content.innerHTML = '<div class="loading">Searching through all verses</div>';

        // Calculate similarity with all verses (instant because embeddings are pre-computed!)
        const similarities = allVersesWithEmbeddings
            .filter(verse => verse.embedding && Array.isArray(verse.embedding))
            .map(verse => ({
                verse: verse,
                relevance: cosineSimilarity(queryEmbedding, verse.embedding)
            }));

        console.log(`✓ Calculated ${similarities.length} similarities`);

        // Sort by relevance and get top results
        similarities.sort((a, b) => b.relevance - a.relevance);
        
        const topResults = similarities.slice(0, topK);
        console.log('Top result relevance:', topResults[0]?.relevance);
        
        return topResults;
        
    } catch (error) {
        console.error('Semantic search error:', error);
        throw error;
    }
}

// Display list of all Surahs
function displaySurahList() {
    currentView = 'home';
    const content = document.getElementById('content');
    
    let html = '<div class="surah-list">';
    
    quranData.surahs.forEach((surah, index) => {
        html += `
            <div class="surah-card" onclick="loadSurah(${index})">
                <div class="surah-number">${index + 1}</div>
                <div class="surah-name">${surah.name}</div>
                <div class="surah-english">${surah.transliteration}</div>
                <div class="surah-info">${surah.translation} • ${surah.total_verses} verses</div>
            </div>
        `;
    });
    
    html += '</div>';
    content.innerHTML = html;
}

// Load a specific Surah
function loadSurah(surahIndex) {
    const surah = quranData.surahs[surahIndex];
    currentView = 'surah';
    const content = document.getElementById('content');
    
    let html = `
        <button class="back-btn" onclick="displaySurahList()">← Back to Surahs</button>
        <div class="surah-header">
            <div class="surah-title">${surah.name} - ${surah.transliteration}</div>
            <div class="surah-meta">
                ${surah.translation} • 
                ${surah.type.charAt(0).toUpperCase() + surah.type.slice(1)} • 
                ${surah.total_verses} verses
            </div>
        </div>
    `;
    
    surah.verses.forEach((verse) => {
        html += `
            <div class="verse">
                <div class="verse-number">Verse ${verse.id}</div>
                <div class="arabic">${verse.text}</div>
                <div class="translation">${verse.translation}</div>
            </div>
        `;
    });
    
    content.innerHTML = html;
}

// Main search function
async function searchVerses() {
    const searchTerm = document.getElementById('searchInput').value.trim();
    const searchMode = document.querySelector('input[name="searchMode"]:checked').value;
    
    if (!searchTerm) {
        alert('Please enter a search term');
        return;
    }

    const searchBtn = document.getElementById('searchBtn');
    searchBtn.disabled = true;
    searchBtn.textContent = 'Searching...';
    
    try {
        if (searchMode === 'semantic') {
            if (!modelLoaded) {
                alert('AI model is still loading. Please wait a moment or use keyword search.');
                return;
            }
            await semanticSearchAndDisplay(searchTerm);
        } else {
            await keywordSearchAndDisplay(searchTerm);
        }
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = 'Search';
    }
}

// Keyword search
async function keywordSearchAndDisplay(searchTerm) {
    const content = document.getElementById('content');
    content.innerHTML = '<div class="loading">Searching...</div>';
    
    const lowerQuery = searchTerm.toLowerCase();
    const matches = allVersesWithEmbeddings.filter(verse => 
        verse.text_translation.toLowerCase().includes(lowerQuery)
    );

    if (matches.length > 0) {
        displayKeywordResults(matches, searchTerm);
    } else {
        content.innerHTML = `
            <div class="error">
                <h3>No results found</h3>
                <p>Try Semantic Search for better results or different keywords</p>
                <button onclick="displaySurahList()" style="margin-top: 15px">Back to Surahs</button>
            </div>
        `;
    }
}

// Semantic search and display
async function semanticSearchAndDisplay(searchTerm) {
    try {
        const results = await semanticSearch(searchTerm, 20);
        
        if (results.length > 0) {
            displaySemanticResults(results, searchTerm);
        } else {
            const content = document.getElementById('content');
            content.innerHTML = `
                <div class="error">
                    <h3>No results found</h3>
                    <p>Try keyword search or different terms</p>
                    <button onclick="displaySurahList()" style="margin-top: 15px">Back to Surahs</button>
                </div>
            `;
        }
    } catch (error) {
        showError(`AI search failed: ${error.message}`);
    }
}

// Display semantic search results
function displaySemanticResults(results, searchTerm) {
    const content = document.getElementById('content');
    
    let html = `
        <button class="back-btn" onclick="displaySurahList()">← Back to Surahs</button>
        <div class="surah-header">
            <div class="surah-title">Semantic Search Results <span class="ai-badge"></span></div>
            <div class="surah-meta">Found ${results.length} verses related to "${searchTerm}"</div>
        </div>
    `;
    
    results.forEach(result => {
        const verse = result.verse;
        const relevance = (result.relevance * 100).toFixed(0);
        
        html += `
            <div class="verse">
                <div class="relevance-score">${relevance}% relevant</div>
                <div class="verse-number">
                    ${verse.surah_english} - Verse ${verse.verse_id}
                </div>
                <div class="arabic">${verse.text_arabic}</div>
                <div class="translation">${verse.text_translation}</div>
            </div>
        `;
    });
    
    content.innerHTML = html;
}

// Display keyword search results
function displayKeywordResults(matches, searchTerm) {
    const content = document.getElementById('content');
    
    let html = `
        <button class="back-btn" onclick="displaySurahList()">← Back to Surahs</button>
        <div class="surah-header">
            <div class="surah-title">Keyword Search Results</div>
            <div class="surah-meta">Found ${matches.length} verses for "${searchTerm}"</div>
        </div>
    `;
    
    matches.slice(0, 50).forEach(verse => {
        html += `
            <div class="verse">
                <div class="verse-number">
                    ${verse.surah_english} - Verse ${verse.verse_id}
                </div>
                <div class="arabic">${verse.text_arabic}</div>
                <div class="translation">${verse.text_translation}</div>
            </div>
        `;
    });
    
    content.innerHTML = html;
}

// Show error message
function showError(message) {
    const content = document.getElementById('content');
    content.innerHTML = `
        <div class="error">
            <h3>Error</h3>
            <p>${message}</p>
            <button onclick="displaySurahList()" style="margin-top: 15px">Back to Surahs</button>
        </div>
    `;
}

// Make functions globally accessible
window.searchVerses = searchVerses;
window.displaySurahList = displaySurahList;
window.loadSurah = loadSurah;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);