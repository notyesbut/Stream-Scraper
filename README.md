# Stream Scraper

Stream Scraper is a tool designed to find and analyze streaming videos on websites. It automatically crawls websites, finds links to video streams, and extracts information about the video quality.

## Functionality

-   **Web Page Scraping**: Automatically crawls websites to find streams and video.
-   **Stream Analysis**: Extracts video quality information (480p, 720p, 1080p, 1440p, 2160p) from HLS and DASH playlists.
-   **Database Storage**: Saves found links and quality information to an SQLite database.
-   **Graphical User Interface**: Intuitive GUI for managing the scraping process and viewing results.
-   **Logging**: Detailed logging of all actions and errors.
-   **Parallel Processing**: Supports multithreading to speed up the scraping process.

## Installation

1.  **Clone the repository**:
    ```
    git clone 
    cd stream-scraper
    ```
2.  **Install the dependencies**:
    ```
    pip install -r requirements.txt
    ```
3.  **Install Playwright browsers**:
    ```
    playwright install
    ```

## Usage

### Command Line

1.  **Single website**:
    ```
    python your_script_name.py [https://www.example.com](https://www.example.com)
    ```
2.  **Multiple websites**:
    ```
    python your_script_name.py [https://www.example.com](https://www.example.com) [invalid URL removed]
    ```

### Graphical User Interface

1.  Run the script:
    ```
    python your_script_name.py
    ```
2.  Enter the website URLs in the "URL" field.
3.  Adjust settings (search depth, number of threads).
4.  Click the "Start" button.
5.  View the results in the "Database" tab.

## Code Description

-   **Logging Setup**: Logging of actions and errors for debugging.
-   **Helper Functions**:
    -   `clean_url`: URL normalization.
    -   `is_stream_url`: Identification of stream links.
    -   `is_placeholder_link`: Placeholder link checking.
    -   `decode_js_escapes`: Decoding JavaScript escape sequences.
    -   `deep_search_for_urls`: Recursive URL search in JSON.
-   **Playlist Parsing Classes**:
    -   `HLSPlaylistParser`: Parsing HLS (.m3u8) playlists.
    -   `DASHPlaylistParser`: Parsing DASH (.mpd) playlists.
-   **Functions for Working with Playlists**:
    -   `fetch_content`: Loading content by URL.
    -   `analyze_playlist`: Playlist analysis.
    -   `determine_playlist_type`: Determining playlist type.
    -   `resolve_url`: Converting a relative URL to an absolute one.
-   **Functions for Working with the Database**:
    -   `init_db`: Initializing the SQLite database.
    -   `update_stream_quality`: Updating stream quality.
    -   `get_stream_by_main_url`: Getting a stream by main_url.
    -   `save_to_db`: Saving stream links to the database.
-   **`NetworkWatcher` Class**: Tracking network requests to find stream links.
-   **`PageParser` Class**: Parsing page content (HTML, video/source, script).
-   **`click_play_buttons` Function**: Simulating "Play" button clicks.
-   **`SiteCrawler` Class**: Managing the crawling process.
-   **`is_url_valid` Function**: Checking URL availability.
-   **`main` Function**: Main function for running the crawler via the command line.
-   **`CrawlerGUI` Class**: Graphical user interface.

## Requirements

-   Python 3.7+
-   aiosqlite
-   websockets
-   playwright
-   beautifulsoup4
-   lxml
-   m3u8
-   xml.etree.ElementTree
-   tkinter
-   urllib
-   re

## License

This project is licensed under the **Custom Non-Commercial License**.

**Custom Non-Commercial License**

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

**Usage Restrictions:**

1.  **Non-Commercial Use Only:** This software may only be used for non-commercial purposes. Non-commercial use means that you may not use the software, or any derivative works, in any way that is primarily intended for or directed toward commercial advantage or monetary compensation.
2.  **No Commercial Distribution:** You may not distribute, sublicense, sell, lease, rent, or otherwise transfer the software, or any derivative works, to any third party for commercial purposes.
3.  **Attribution:** You must retain all copyright, trademark, and other proprietary notices contained in the original software and provide attribution to the original author in any derivative works.
4.  **Prior Written Consent for Commercial Use:** Any use of this software for commercial purposes requires the prior written consent of the copyright holder. To request permission for commercial use, please contact qvzzcb@gmail.com.
5. **Disclaimer:** The author shall not be held liable for any damages arising from the use of this software, even if advised of the possibility of such damages.

**Termination:**
This license automatically terminates if you violate any of its terms. Upon termination, you must cease all use of the software and destroy all copies, full or partial, of the software.

## Author

qvzzcb@gmail.com
