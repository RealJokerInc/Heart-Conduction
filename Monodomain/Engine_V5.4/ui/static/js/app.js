/**
 * Builder UI - Client-side JavaScript
 */

// Utility functions
const Utils = {
    async fetchJSON(url, options = {}) {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });
        return response.json();
    },

    async postJSON(url, data) {
        return this.fetchJSON(url, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },
};

// Export for use in templates
window.Utils = Utils;
