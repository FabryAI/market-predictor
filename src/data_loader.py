import pandas as pd
from pathlib import Path

class PriceDataLoader:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Carica il file CSV e converte il timestamp in datetime.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File non trovato: {self.filepath}")

        df = pd.read_csv(self.filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.rename(columns={'Timestamp': 'Datetime'})
        df.set_index('Datetime', inplace=True)
        self.df = df

        return df

    def get_summary(self) -> str:
        """
        Restituisce un riepilogo base del dataset caricato.
        """
        if self.df is None:
            raise ValueError("I dati non sono stati caricati. Chiama prima load_data().")

        return str(self.df.describe())

    def save_processed(self, output_path: str):
        """
        Salva i dati elaborati come CSV.
        """
        if self.df is None:
            raise ValueError("I dati non sono stati caricati o elaborati.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_file)
