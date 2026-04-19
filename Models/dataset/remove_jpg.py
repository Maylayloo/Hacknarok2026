import os


def clean_images_from_dataset(dataset_path="dataset-kopia"):
    # Rozszerzenia plików, które chcemy usunąć (możesz dodać .png itp., jeśli masz inne)
    extensions_to_delete = ('.jpg', '.jpeg', '.png')

    if not os.path.exists(dataset_path):
        print(f"Błąd: Folder '{dataset_path}' nie istnieje w obecnej ścieżce.")
        return

    print(f"Rozpoczynam czyszczenie folderu: {dataset_path}...")
    deleted_count = 0

    # os.walk() przelatuje przez wszystkie foldery i podfoldery
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Sprawdzamy, czy plik kończy się na jedno z wybranych rozszerzeń
            if file.lower().endswith(extensions_to_delete):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Nie udało się usunąć pliku {file_path}. Błąd: {e}")

    print("-" * 30)
    print(f"Gotowe! Usunięto łącznie {deleted_count} plików graficznych.")
    print("Pliki .json pozostały nienaruszone.")


if __name__ == "__main__":
    # Upewnij się dwa razy, że odpalasz to na kopii!
    clean_images_from_dataset("dataset")