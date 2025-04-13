#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Terminal-Ausgabe-Hilfsfunktionen für zuverlässige Konsolenausgabe
in verschiedenen Umgebungen (Windows, IDE, Cascade, etc.).
"""

import os
import sys
import time
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


def ensure_output_display(command: str, cwd: str = None, shell: bool = True) -> str:
    """
    Führt einen Befehl aus und stellt sicher, dass die Ausgabe korrekt angezeigt wird,
    indem die Ausgabe in eine Datei umgeleitet und dann gelesen wird.
    
    Args:
        command: Der auszuführende Befehl
        cwd: Das Arbeitsverzeichnis (optional)
        shell: Ob der Befehl in einer Shell ausgeführt werden soll
        
    Returns:
        Der Inhalt der Ausgabe als String
    """
    # Timestamp für eindeutige Dateinamen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Temporäre Ausgabedatei erstellen
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"cmd_output_{timestamp}.txt"
    
    # Befehl mit Umleitung
    redirected_command = f"{command} > {output_file} 2>&1"
    
    # Befehl ausführen
    print(f"Führe Befehl aus: {command}")
    print(f"Ausgabe wird nach {output_file} umgeleitet")
    
    exit_code = subprocess.call(redirected_command, cwd=cwd, shell=shell)
    
    # Warten, bis die Datei existiert und vollständig geschrieben ist
    max_wait = 5  # Maximale Wartezeit in Sekunden
    waited = 0
    while not output_file.exists() and waited < max_wait:
        time.sleep(0.1)
        waited += 0.1
    
    # Wenn die Datei existiert, Inhalt lesen
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            output = f.read()
        
        # Ausgabe anzeigen
        print("\n" + "="*80)
        print(f"BEFEHL: {command}")
        print(f"EXITCODE: {exit_code}")
        print("="*80)
        print(output)
        print("="*80 + "\n")
        
        return output
    else:
        message = f"Konnte keine Ausgabe erfassen für Befehl: {command}"
        print(message)
        return message


def run_python_script(script_path: str, args: str = "", cwd: str = None) -> str:
    """
    Führt ein Python-Skript aus und stellt sicher, dass die Ausgabe angezeigt wird.
    
    Args:
        script_path: Pfad zum Python-Skript
        args: Zusätzliche Argumente für das Skript (optional)
        cwd: Das Arbeitsverzeichnis (optional)
        
    Returns:
        Der Inhalt der Ausgabe als String
    """
    # Python-Befehl mit -u für ungepufferte Ausgabe
    python_cmd = sys.executable
    command = f"{python_cmd} -u {script_path} {args}"
    
    return ensure_output_display(command, cwd)


def run_python_code(code: str, args: str = "", cwd: str = None) -> str:
    """
    Führt Python-Code als String aus und stellt sicher, dass die Ausgabe angezeigt wird.
    
    Args:
        code: Python-Code als String
        args: Zusätzliche Argumente (optional)
        cwd: Das Arbeitsverzeichnis (optional)
        
    Returns:
        Der Inhalt der Ausgabe als String
    """
    # Temporäre Python-Datei erstellen
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        temp_script = f.name
    
    try:
        # Skript ausführen
        return run_python_script(temp_script, args, cwd)
    finally:
        # Temporäre Datei löschen
        try:
            os.unlink(temp_script)
        except:
            pass


if __name__ == "__main__":
    # Beispielaufruf
    print("Terminal-Ausgabe-Test")
    
    # Beispiel 1: Befehl ausführen
    output = ensure_output_display("dir")
    print(f"Ausgabezeilen: {len(output.splitlines())}")
    
    # Beispiel 2: Python-Skript ausführen
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        output = run_python_script(script_path)
        print(f"Skript-Ausgabezeilen: {len(output.splitlines())}")
