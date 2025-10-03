// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recommendation-form');
    const bookSearchInput = document.getElementById('book-search');
    const searchResultsDiv = document.getElementById('search-results');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const cfWeightSlider = document.getElementById('cf-weight');
    const weightLabel = document.getElementById('weight-label');

    let selectedBookId = null; // To store the book_id of the selected book

    // --- Event Listeners ---
    form.addEventListener('submit', handleFormSubmit);
    bookSearchInput.addEventListener('input', handleBookSearch);
    cfWeightSlider.addEventListener('input', updateWeightLabel);

    // --- Functions ---
    function updateWeightLabel() {
        const value = cfWeightSlider.value;
        if (value < 40) {
            weightLabel.textContent = `Content (${100 - value}%)`;
        } else {
            weightLabel.textContent = `Community (${value}%)`;
        }
    }

    async function handleBookSearch(event) {
        const query = event.target.value;
        if (query.length < 3) {
            searchResultsDiv.innerHTML = '';
            searchResultsDiv.classList.remove('active');
            return;
        }

        try {
            const response = await fetch(`/books/search?q=${encodeURIComponent(query)}&limit=5`);
            const books = await response.json();

            searchResultsDiv.innerHTML = '';
            if (books.length > 0) {
                books.forEach(book => {
                    const div = document.createElement('div');
                    div.classList.add('search-result-item');
                    div.textContent = book.title;
                    div.addEventListener('click', () => selectBook(book.title, book.book_id));
                    searchResultsDiv.appendChild(div);
                });
            } else {
                searchResultsDiv.innerHTML = '<div class="search-result-item">No results found</div>';
            }
        } catch (error) {
            console.error('Error searching for books:', error);
            searchResultsDiv.innerHTML = '<div class="search-result-item">Search failed</div>';
        }
    }

    function selectBook(title, bookId) {
        bookSearchInput.value = title;
        selectedBookId = bookId;
        searchResultsDiv.innerHTML = '';
        searchResultsDiv.classList.remove('active');
    }

    async function handleFormSubmit(event) {
        event.preventDefault();
        
        const userId = document.getElementById('user-id').value;
        const likedBookTitle = bookSearchInput.value;
        const cfWeight = cfWeightSlider.value / 100;

        if (!userId || !likedBookTitle) {
            alert('Please fill in all fields.');
            return;
        }

        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        recommendationsContainer.classList.add('hidden');

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: parseInt(userId),
                    liked_book_title: likedBookTitle,
                    cf_weight: cfWeight
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get recommendations.');
            }

            const data = await response.json();
            displayRecommendations(data.recommendations);

        } catch (error) {
            console.error('Error getting recommendations:', error);
            alert(error.message);
        } finally {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
        }
    }

    function displayRecommendations(recommendations) {
        recommendationsContainer.innerHTML = ''; // Clear previous results
        const grid = document.createElement('div');
        grid.classList.add('recommendations-grid');

        if (recommendations.length === 0) {
            recommendationsContainer.innerHTML = '<p>No recommendations found. Try a different book or user.</p>';
        } else {
            recommendations.forEach(book => {
                const card = document.createElement('div');
                card.classList.add('book-card');
                card.innerHTML = `
                    <img src="${book.image_url}" alt="${book.title}">
                    <h3>${book.title}</h3>
                    <p>by ${book.authors}</p>
                `;
                grid.appendChild(card);
            });
            recommendationsContainer.appendChild(grid);
        }
        
        recommendationsContainer.classList.remove('hidden');
    }
});