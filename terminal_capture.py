#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Terminal-Ausgabe-Capture-Utility für das POCO Dashboard.

Diese Utility stellt die zuverlässige Erfassung und Anzeige von Terminalausgaben sicher,
unabhängig von Buffering-Problemen oder IDE-Einschränkungen.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any


class TerminalCapture:
    """
    Klasse zur zuverlässigen Erfassung und Protokollierung von Terminalausgaben.
    
    Diese Klasse löst das Problem von Buffering und verschwundener Ausgabe in verschiedenen
    Umgebungen, indem sie alle Ausgaben in Dateien umleitet und strukturiert erfasst.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        prefix: str = "terminal_output",
        log_level: int = logging.INFO
    ):
        """
        Initialisiert den TerminalCapture.
        
        Args:
            log_dir: Verzeichnis für Ausgabedateien
            prefix: Präfix für Ausgabedateinamen
            log_level: Logging-Level (standardmäßig INFO)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        
        # Logger einrichten
        self.logger = logging.getLogger(f"terminal_capture_{prefix}")
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            # Konsolen-Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # Formatter
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
    
    def run_command(
        self,
        command: Union[str, List[str]],
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = True,
        timeout: Optional[int] = None,
        description: Optional[str] = None
    ) -> Tuple[int, str, Path]:
        """
        Führt einen Befehl aus und erfasst die Ausgabe zuverlässig.
        
        Args:
            command: Der auszuführende Befehl (String oder Liste)
            cwd: Arbeitsverzeichnis für die Ausführung
            env: Umgebungsvariablen für die Ausführung
            shell: Ob der Befehl in einer Shell ausgeführt werden soll
            timeout: Timeout in Sekunden (optional)
            description: Beschreibung des Befehls für Logging
            
        Returns:
            Tuple mit (Exit-Code, Ausgabe als String, Pfad zur Ausgabedatei)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.log_dir / f"{self.prefix}_{timestamp}.txt"
        
        # Befehl für die Protokollierung formatieren
        if isinstance(command, list):
            cmd_str = " ".join(command)
        else:
            cmd_str = command
        
        # Befehlsbeschreibung für die Protokollierung
        cmd_desc = description or f"Befehl: {cmd_str[:80]}..."
        
        self.logger.info(f"Führe {cmd_desc} aus")
        self.logger.info(f"Ausgabe wird nach {output_file} umgeleitet")
        
        # Ausgabedatei-Header schreiben
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"=== Terminal-Ausgabe-Erfassung ===\n")
            f.write(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Befehl: {cmd_str}\n")
            f.write(f"Arbeitsverzeichnis: {cwd or os.getcwd()}\n")
            f.write("="*80 + "\n\n")
        
        # Befehl mit Umleitung
        redirected_cmd = f"{cmd_str} >> \"{output_file}\" 2>&1"
        
        # Umgebungsvariablen vorbereiten
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        
        # Buffering-Probleme umgehen
        merged_env["PYTHONUNBUFFERED"] = "1"
        
        start_time = time.time()
        
        try:
            # Befehl ausführen
            exit_code = subprocess.call(
                redirected_cmd,
                cwd=cwd,
                shell=shell,
                env=merged_env,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Dauer in Ausgabedatei schreiben
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== Befehl abgeschlossen ===\n")
                f.write(f"Exit-Code: {exit_code}\n")
                f.write(f"Dauer: {duration:.2f} Sekunden\n")
                f.write(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"Befehl abgeschlossen mit Exit-Code {exit_code} in {duration:.2f} Sekunden")
            
            # Ausgabe lesen
            with open(output_file, "r", encoding="utf-8") as f:
                output = f.read()
            
            return exit_code, output, output_file
            
        except subprocess.TimeoutExpired:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== TIMEOUT nach {timeout} Sekunden ===\n")
            
            self.logger.error(f"Timeout nach {timeout} Sekunden")
            
            # Ausgabe bis zum Timeout lesen
            with open(output_file, "r", encoding="utf-8") as f:
                output = f.read()
            
            return 124, output, output_file  # 124 ist der Standard-Exit-Code für Timeout
            
        except Exception as e:
            error_msg = f"Fehler bei der Ausführung: {str(e)}"
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== FEHLER ===\n")
                f.write(f"{error_msg}\n")
            
            self.logger.error(error_msg)
            
            # Ausgabe bis zum Fehler lesen
            with open(output_file, "r", encoding="utf-8") as f:
                output = f.read()
            
            return 1, output, output_file
    
    def run_python_script(
        self,
        script_path: Union[str, Path],
        args: Optional[Union[str, List[str]]] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        description: Optional[str] = None
    ) -> Tuple[int, str, Path]:
        """
        Führt ein Python-Skript aus und erfasst die Ausgabe zuverlässig.
        
        Args:
            script_path: Pfad zum Python-Skript
            args: Argumente für das Skript (String oder Liste)
            cwd: Arbeitsverzeichnis für die Ausführung
            env: Umgebungsvariablen für die Ausführung
            timeout: Timeout in Sekunden (optional)
            description: Beschreibung des Skripts für Logging
            
        Returns:
            Tuple mit (Exit-Code, Ausgabe als String, Pfad zur Ausgabedatei)
        """
        script_path = Path(script_path)
        
        # Argumente formatieren
        if args is None:
            args_str = ""
        elif isinstance(args, list):
            args_str = " ".join(args)
        else:
            args_str = args
        
        # Python-Befehl mit -u für ungepufferte Ausgabe
        python_cmd = f"{sys.executable} -u {script_path} {args_str}"
        
        # Beschreibung für Logging
        script_desc = description or f"Python-Skript: {script_path.name}"
        
        return self.run_command(
            python_cmd,
            cwd=cwd,
            env=env,
            shell=True,
            timeout=timeout,
            description=script_desc
        )
    
    def run_python_code(
        self,
        code: str,
        args: Optional[Union[str, List[str]]] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        description: Optional[str] = None
    ) -> Tuple[int, str, Path]:
        """
        Führt Python-Code aus und erfasst die Ausgabe zuverlässig.
        
        Args:
            code: Python-Code als String
            args: Argumente für das Skript (String oder Liste)
            cwd: Arbeitsverzeichnis für die Ausführung
            env: Umgebungsvariablen für die Ausführung
            timeout: Timeout in Sekunden (optional)
            description: Beschreibung des Codes für Logging
            
        Returns:
            Tuple mit (Exit-Code, Ausgabe als String, Pfad zur Ausgabedatei)
        """
        # Temporäre Python-Datei erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_script = self.log_dir / f"temp_script_{timestamp}.py"
        
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(code)
        
        try:
            # Code ausführen
            return self.run_python_script(
                temp_script,
                args=args,
                cwd=cwd,
                env=env,
                timeout=timeout,
                description=description or "Temporärer Python-Code"
            )
        finally:
            # Temporäre Datei löschen
            try:
                os.unlink(temp_script)
            except:
                pass
    
    def display_file_content(self, file_path: Union[str, Path], max_lines: Optional[int] = None) -> str:
        """
        Zeigt den Inhalt einer Datei an und gibt ihn zurück.
        
        Args:
            file_path: Pfad zur Datei
            max_lines: Maximale Anzahl von Zeilen, die angezeigt werden sollen
            
        Returns:
            Dateiinhalt als String
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            message = f"Datei {file_path} existiert nicht"
            self.logger.error(message)
            return message
        
        self.logger.info(f"Zeige Inhalt von {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            if max_lines is None:
                content = f.read()
            else:
                content = "".join(f.readlines()[:max_lines])
                if len(content.splitlines()) >= max_lines:
                    content += f"\n... [gekürzt, zeige {max_lines} von {sum(1 for _ in open(file_path, 'r'))} Zeilen]"
        
        print("\n" + "="*80)
        print(f"DATEIINHALT: {file_path}")
        print("="*80)
        print(content)
        print("="*80 + "\n")
        
        return content


# Singleton-Instanz für einfachen globalen Zugriff
default_capture = TerminalCapture()

# Komfortfunktionen für direkten Zugriff
def run_command(
    command: Union[str, List[str]],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    shell: bool = True,
    timeout: Optional[int] = None,
    description: Optional[str] = None
) -> Tuple[int, str, Path]:
    """
    Führt einen Befehl aus und erfasst die Ausgabe zuverlässig.
    
    Args:
        command: Der auszuführende Befehl (String oder Liste)
        cwd: Arbeitsverzeichnis für die Ausführung
        env: Umgebungsvariablen für die Ausführung
        shell: Ob der Befehl in einer Shell ausgeführt werden soll
        timeout: Timeout in Sekunden (optional)
        description: Beschreibung des Befehls für Logging
        
    Returns:
        Tuple mit (Exit-Code, Ausgabe als String, Pfad zur Ausgabedatei)
    """
    return default_capture.run_command(command, cwd, env, shell, timeout, description)

def run_python_script(
    script_path: Union[str, Path],
    args: Optional[Union[str, List[str]]] = None,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    description: Optional[str] = None
) -> Tuple[int, str, Path]:
    """
    Führt ein Python-Skript aus und erfasst die Ausgabe zuverlässig.
    
    Args:
        script_path: Pfad zum Python-Skript
        args: Argumente für das Skript (String oder Liste)
        cwd: Arbeitsverzeichnis für die Ausführung
        env: Umgebungsvariablen für die Ausführung
        timeout: Timeout in Sekunden (optional)
        description: Beschreibung des Skripts für Logging
        
    Returns:
        Tuple mit (Exit-Code, Ausgabe als String, Pfad zur Ausgabedatei)
    """
    return default_capture.run_python_script(script_path, args, cwd, env, timeout, description)

def run_python_code(
    code: str,
    args: Optional[Union[str, List[str]]] = None,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    description: Optional[str] = None
) -> Tuple[int, str, Path]:
    """
    Führt Python-Code aus und erfasst die Ausgabe zuverlässig.
    
    Args:
        code: Python-Code als String
        args: Argumente für das Skript (String oder Liste)
        cwd: Arbeitsverzeichnis für die Ausführung
        env: Umgebungsvariablen für die Ausführung
        timeout: Timeout in Sekunden (optional)
        description: Beschreibung des Codes für Logging
        
    Returns:
        Tuple mit (Exit-Code, Ausgabe als String, Pfad zur Ausgabedatei)
    """
    return default_capture.run_python_code(code, args, cwd, env, timeout, description)


if __name__ == "__main__":
    # Beispielverwendung
    print("Terminal-Capture-Test")
    
    # Beispiel 1: Standardbefehl
    exit_code, output, log_file = run_command("dir")
    print(f"Exit-Code: {exit_code}")
    print(f"Ausgabedatei: {log_file}")
    
    # Beispiel 2: Python-Code
    code = """
import sys
import os

print("Python-Umgebung:")
print(f"Python-Version: {sys.version}")
print(f"Arbeitsverzeichnis: {os.getcwd()}")
print(f"Umgebungsvariablen: {dict(os.environ)}")
"""
    
    exit_code, output, log_file = run_python_code(code, description="Python-Umgebungsinfo")
    print(f"Exit-Code: {exit_code}")
    print(f"Ausgabedatei: {log_file}")
