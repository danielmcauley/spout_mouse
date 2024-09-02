import json
import gspread
from google.oauth2 import service_account
import pandas as pd


def authorize_google_sheets(credentials_json: str) -> gspread.Client:
    """
    Authorize access to Google Sheets using provided credentials.

    Args:
        credentials_json (str): JSON string of Google service account credentials.

    Returns:
        gspread.Client: Authorized Google Sheets client.

    Raises:
        ValueError: If the credentials JSON is invalid.
        ConnectionError: If authorization fails due to connection issues.
    """
    try:
        service_account_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        scoped_credentials = credentials.with_scopes([
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ])
        client = gspread.authorize(scoped_credentials)
        return client
    except json.JSONDecodeError as e:
        raise ValueError("Invalid credentials JSON provided.") from e
    except Exception as e:
        raise ConnectionError("Failed to authorize Google Sheets client.") from e


def get_experiment_data(client: gspread.Client, google_sheet_url: str, worksheet_index: int = 0) -> pd.DataFrame:
    """
    Fetch experiment data from a Google Sheets document.

    Args:
        client (gspread.Client): Authorized Google Sheets client.
        google_sheet_url (str, optional): URL of the Google Sheets document.
        worksheet_index (int, optional): Index of the worksheet to retrieve. Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame containing experiment records.

    Raises:
        ValueError: If fetching data fails.
    """
    try:
        sheet_url = google_sheet_url
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.get_worksheet(worksheet_index)
        records = worksheet.get_all_records()
        data = pd.DataFrame(records)
        return data
    except Exception as e:
        raise ValueError("Failed to fetch data from Google Sheets.") from e

