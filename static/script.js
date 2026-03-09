// script.js — shared utilities (page-specific logic lives in inline <script> blocks)

// Auto-dismiss flash messages after 5 seconds
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.flash').forEach(flash => {
        setTimeout(() => flash.remove(), 5000);
    });
});
