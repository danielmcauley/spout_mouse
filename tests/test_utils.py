import pytest
import pandas as pd
from unittest.mock import Mock
from spout_mouse import utils


def test_authorize_google_sheets_success(mocker):
    # Mock credentials JSON
    credentials_json = '{"type": "service_account", "project_id": "test_project", "private_key_id": "test_key_id", "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEv...", "client_email": "test@test-project.iam.gserviceaccount.com", "client_id": "1234567890", "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test@test-project.iam.gserviceaccount.com"}'
    
    # Create mock credentials object
    mock_credentials = Mock()
    mock_scoped_credentials = Mock()
    mock_google_sheet_url = Mock()
    
    # Mock the from_service_account_info method
    mocker.patch(
        'google.oauth2.service_account.Credentials.from_service_account_info',
        return_value=mock_credentials
    )
    
    # Mock the with_scopes method on the credentials
    mock_credentials.with_scopes.return_value = mock_scoped_credentials
    
    # Mock gspread.authorize
    mock_gspread_authorize = mocker.patch(
        'gspread.authorize',
        return_value=Mock()
    )
    
    # Call the function under test
    client = utils.authorize_google_sheets(credentials_json)
    
    # Assertions
    assert client is not None
    mock_credentials.with_scopes.assert_called_once_with([
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ])
    mock_gspread_authorize.assert_called_once_with(mock_scoped_credentials)



def test_authorize_google_sheets_invalid_json():
    credentials_json = 'invalid json'
    with pytest.raises(ValueError):
        utils.authorize_google_sheets(credentials_json)


def test_get_experiment_data_success(mocker):
    mock_client = Mock()
    mock_sheet = Mock()
    mock_worksheet = Mock()
    mock_worksheet.get_all_records.return_value = [
        {'experiment': 'test_exp', 'cohort': 1, 'day': 1, 'spout_id': 1, 'spout_name': 'water'}
    ]
    mock_sheet.get_worksheet.return_value = mock_worksheet
    mock_client.open_by_url.return_value = mock_sheet
    
    data = utils.get_experiment_data(mock_client, mock_sheet)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert list(data.columns) == ['experiment', 'cohort', 'day', 'spout_id', 'spout_name']


def test_get_experiment_data_failure(mocker):
    mock_client = Mock()
    mock_sheet = Mock()
    mock_client.open_by_url.side_effect = Exception("Connection error")
    
    with pytest.raises(ValueError):
        utils.get_experiment_data(mock_client, mock_sheet)
