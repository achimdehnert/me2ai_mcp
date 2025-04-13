"""
Tests for the ME2AI MCP filesystem tools.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, mock_open
import os
import base64

from me2ai_mcp.tools.filesystem import FileReaderTool, FileWriterTool, DirectoryListerTool


class TestFileReaderTool:
    """Tests for the FileReaderTool class."""

    def test_init(self):
        """Test tool initialization."""
        tool = FileReaderTool(
            name="custom-reader",
            description="Custom file reader",
            max_file_size=1024 * 1024  # 1MB
        )
        
        assert tool.name == "custom-reader"
        assert tool.description == "Custom file reader"
        assert tool.max_file_size == 1024 * 1024
        
    @pytest.mark.asyncio
    async def test_missing_file_path(self):
        """Test file read with missing file path."""
        tool = FileReaderTool()
        
        result = await tool.execute({})
        
        assert result["success"] is False
        assert "file_path parameter is required" in result["error"]
        
    @pytest.mark.asyncio
    async def test_file_not_found(self):
        """Test file read when file doesn't exist."""
        tool = FileReaderTool()
        
        with patch('os.path.exists', return_value=False):
            result = await tool.execute({"file_path": "/path/to/nonexistent/file.txt"})
            
            assert result["success"] is False
            assert "File not found" in result["error"]
            
    @pytest.mark.asyncio
    async def test_path_not_file(self):
        """Test file read when path is not a file."""
        tool = FileReaderTool()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=False):
                result = await tool.execute({"file_path": "/path/to/directory"})
                
                assert result["success"] is False
                assert "Path is not a file" in result["error"]
                
    @pytest.mark.asyncio
    async def test_file_too_large(self):
        """Test file read when file is too large."""
        tool = FileReaderTool(max_file_size=100)  # 100 bytes max
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.getsize', return_value=1000):  # 1000 bytes
                    result = await tool.execute({"file_path": "/path/to/large_file.txt"})
                    
                    assert result["success"] is False
                    assert "File too large" in result["error"]
                    
    @pytest.mark.asyncio
    async def test_read_text_file(self):
        """Test reading a text file."""
        tool = FileReaderTool()
        file_content = "This is test content"
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.getsize', return_value=len(file_content)):
                    with patch('os.path.abspath', return_value="/abs/path/to/file.txt"):
                        with patch('builtins.open', mock_open(read_data=file_content)) as mock_file:
                            with patch('os.stat') as mock_stat:
                                # Configure mock stat
                                mock_stat_result = MagicMock()
                                mock_stat_result.st_ctime = 1000
                                mock_stat_result.st_mtime = 2000
                                mock_stat_result.st_atime = 3000
                                mock_stat.return_value = mock_stat_result
                                
                                result = await tool.execute({
                                    "file_path": "file.txt",
                                    "encoding": "utf-8"
                                })
                                
                                assert result["success"] is True
                                assert result["file_path"] == "/abs/path/to/file.txt"
                                assert result["content"] == file_content
                                assert result["size"] == len(file_content)
                                assert result["encoding"] == "utf-8"
                                assert result["binary"] is False
                                assert "metadata" in result
                                assert result["metadata"]["created"] == 1000
                                assert result["metadata"]["modified"] == 2000
                                assert result["metadata"]["extension"] == ".txt"
                                
    @pytest.mark.asyncio
    async def test_read_binary_file(self):
        """Test reading a binary file."""
        tool = FileReaderTool()
        file_content = b"\x00\x01\x02\x03"
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.getsize', return_value=len(file_content)):
                    with patch('os.path.abspath', return_value="/abs/path/to/file.bin"):
                        with patch('builtins.open', mock_open(read_data=file_content)) as mock_file:
                            with patch('os.stat') as mock_stat:
                                # Configure mock stat
                                mock_stat_result = MagicMock()
                                mock_stat_result.st_ctime = 1000
                                mock_stat_result.st_mtime = 2000
                                mock_stat_result.st_atime = 3000
                                mock_stat.return_value = mock_stat_result
                                
                                # Base64 encoded content
                                encoded_content = base64.b64encode(file_content).decode("utf-8")
                                
                                result = await tool.execute({
                                    "file_path": "file.bin",
                                    "binary": True
                                })
                                
                                assert result["success"] is True
                                assert result["file_path"] == "/abs/path/to/file.bin"
                                assert result["content"] == encoded_content
                                assert result["size"] == len(file_content)
                                assert result["encoding"] is None
                                assert result["binary"] is True


class TestFileWriterTool:
    """Tests for the FileWriterTool class."""
    
    @pytest.mark.asyncio
    async def test_missing_file_path(self):
        """Test file write with missing file path."""
        tool = FileWriterTool()
        
        result = await tool.execute({"content": "Test content"})
        
        assert result["success"] is False
        assert "file_path parameter is required" in result["error"]
        
    @pytest.mark.asyncio
    async def test_missing_content(self):
        """Test file write with missing content."""
        tool = FileWriterTool()
        
        result = await tool.execute({"file_path": "/path/to/file.txt"})
        
        assert result["success"] is False
        assert "content parameter is required" in result["error"]
        
    @pytest.mark.asyncio
    async def test_file_exists_no_overwrite(self):
        """Test file write when file exists and overwrite is False."""
        tool = FileWriterTool()
        
        with patch('os.path.exists', return_value=True):
            result = await tool.execute({
                "file_path": "/path/to/file.txt",
                "content": "Test content",
                "overwrite": False
            })
            
            assert result["success"] is False
            assert "File already exists" in result["error"]
            
    @pytest.mark.asyncio
    async def test_write_text_file(self):
        """Test writing a text file."""
        tool = FileWriterTool()
        
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs') as mock_makedirs:
                with patch('os.path.abspath', return_value="/abs/path/to/file.txt"):
                    with patch('builtins.open', mock_open()) as mock_file:
                        with patch('os.path.getsize', return_value=12):
                            with patch('os.stat') as mock_stat:
                                # Configure mock stat
                                mock_stat_result = MagicMock()
                                mock_stat_result.st_ctime = 1000
                                mock_stat_result.st_mtime = 2000
                                mock_stat.return_value = mock_stat_result
                                
                                result = await tool.execute({
                                    "file_path": "/path/to/file.txt",
                                    "content": "Test content",
                                    "encoding": "utf-8"
                                })
                                
                                # Check makedirs was called
                                mock_makedirs.assert_called_once_with("/path/to", exist_ok=True)
                                
                                # Check file was opened correctly
                                mock_file.assert_called_once_with("/abs/path/to/file.txt", "w", encoding="utf-8")
                                
                                # Check content was written
                                mock_file().write.assert_called_once_with("Test content")
                                
                                assert result["success"] is True
                                assert result["file_path"] == "/abs/path/to/file.txt"
                                assert result["size"] == 12
                                assert result["operation"] == "create"
                                
    @pytest.mark.asyncio
    async def test_append_to_file(self):
        """Test appending to a file."""
        tool = FileWriterTool()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs') as mock_makedirs:
                with patch('os.path.abspath', return_value="/abs/path/to/file.txt"):
                    with patch('builtins.open', mock_open()) as mock_file:
                        with patch('os.path.getsize', return_value=24):
                            with patch('os.stat') as mock_stat:
                                # Configure mock stat
                                mock_stat_result = MagicMock()
                                mock_stat_result.st_ctime = 1000
                                mock_stat_result.st_mtime = 2000
                                mock_stat.return_value = mock_stat_result
                                
                                result = await tool.execute({
                                    "file_path": "/path/to/file.txt",
                                    "content": "Appended content",
                                    "append": True
                                })
                                
                                # Check file was opened correctly (append mode)
                                mock_file.assert_called_once_with("/abs/path/to/file.txt", "a", encoding="utf-8")
                                
                                # Check content was written
                                mock_file().write.assert_called_once_with("Appended content")
                                
                                assert result["success"] is True
                                assert result["operation"] == "append"


class TestDirectoryListerTool:
    """Tests for the DirectoryListerTool class."""
    
    @pytest.mark.asyncio
    async def test_missing_directory_path(self):
        """Test directory listing with missing directory path."""
        tool = DirectoryListerTool()
        
        result = await tool.execute({})
        
        assert result["success"] is False
        assert "directory_path parameter is required" in result["error"]
        
    @pytest.mark.asyncio
    async def test_directory_not_found(self):
        """Test directory listing when directory doesn't exist."""
        tool = DirectoryListerTool()
        
        with patch('os.path.exists', return_value=False):
            result = await tool.execute({"directory_path": "/path/to/nonexistent"})
            
            assert result["success"] is False
            assert "Directory not found" in result["error"]
            
    @pytest.mark.asyncio
    async def test_path_not_directory(self):
        """Test directory listing when path is not a directory."""
        tool = DirectoryListerTool()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=False):
                result = await tool.execute({"directory_path": "/path/to/file.txt"})
                
                assert result["success"] is False
                assert "Path is not a directory" in result["error"]
                
    @pytest.mark.asyncio
    async def test_list_directory_non_recursive(self):
        """Test non-recursive directory listing."""
        tool = DirectoryListerTool()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', return_value="/abs/path/to/dir"):
                    with patch('os.listdir', return_value=["file1.txt", "file2.txt", "subdir"]):
                        with patch('os.path.join', side_effect=lambda *args: "/".join(args)):
                            with patch('os.path.isfile', side_effect=lambda path: not path.endswith("subdir")):
                                with patch('os.stat') as mock_stat:
                                    # Configure mock stat
                                    mock_stat_result = MagicMock()
                                    mock_stat_result.st_ctime = 1000
                                    mock_stat_result.st_mtime = 2000
                                    mock_stat_result.st_size = 100
                                    mock_stat.return_value = mock_stat_result
                                    
                                    result = await tool.execute({
                                        "directory_path": "/path/to/dir",
                                        "recursive": False
                                    })
                                    
                                    assert result["success"] is True
                                    assert result["directory_path"] == "/abs/path/to/dir"
                                    assert result["recursive"] is False
                                    assert len(result["items"]) == 3
                                    
                                    # Check files are listed correctly
                                    file_items = [item for item in result["items"] if item["type"] == "file"]
                                    assert len(file_items) == 2
                                    assert any(item["name"] == "file1.txt" for item in file_items)
                                    assert any(item["name"] == "file2.txt" for item in file_items)
                                    
                                    # Check directories are listed correctly
                                    dir_items = [item for item in result["items"] if item["type"] == "directory"]
                                    assert len(dir_items) == 1
                                    assert dir_items[0]["name"] == "subdir"
                                    assert dir_items[0]["size"] is None
                                    
    @pytest.mark.asyncio
    async def test_list_directory_with_pattern(self):
        """Test directory listing with pattern filter."""
        tool = DirectoryListerTool()
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', return_value="/abs/path/to/dir"):
                    with patch('os.listdir', return_value=["file1.txt", "file2.txt", "file.dat", "subdir"]):
                        with patch('os.path.join', side_effect=lambda *args: "/".join(args)):
                            with patch('os.path.isfile', side_effect=lambda path: not path.endswith("subdir")):
                                with patch('glob.fnmatch.fnmatch', side_effect=lambda name, pattern: name.endswith(".txt")):
                                    with patch('os.stat') as mock_stat:
                                        # Configure mock stat
                                        mock_stat_result = MagicMock()
                                        mock_stat_result.st_ctime = 1000
                                        mock_stat_result.st_mtime = 2000
                                        mock_stat_result.st_size = 100
                                        mock_stat.return_value = mock_stat_result
                                        
                                        result = await tool.execute({
                                            "directory_path": "/path/to/dir",
                                            "pattern": "*.txt"
                                        })
                                        
                                        assert result["success"] is True
                                        assert result["pattern"] == "*.txt"
                                        
                                        # Only .txt files and the directory should be included
                                        assert len(result["items"]) == 3
                                        file_names = [item["name"] for item in result["items"] if item["type"] == "file"]
                                        assert set(file_names) == {"file1.txt", "file2.txt"}
                                        
    @pytest.mark.asyncio
    async def test_list_directory_recursive(self):
        """Test recursive directory listing."""
        tool = DirectoryListerTool()
        
        # Mock os.walk to return a directory structure
        mock_walk_result = [
            ("/path/to/dir", ["subdir1", "subdir2"], ["file1.txt", "file2.txt"]),
            ("/path/to/dir/subdir1", [], ["file3.txt"]),
            ("/path/to/dir/subdir2", ["subdir3"], ["file4.txt"]),
            ("/path/to/dir/subdir2/subdir3", [], ["file5.txt"])
        ]
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('os.path.abspath', return_value="/path/to/dir"):
                    with patch('os.walk', return_value=mock_walk_result):
                        with patch('os.path.join', side_effect=lambda *args: "/".join(args)):
                            with patch('os.path.relpath', side_effect=lambda path, start: path.replace(start + "/", "")):
                                with patch('os.stat') as mock_stat:
                                    # Configure mock stat
                                    mock_stat_result = MagicMock()
                                    mock_stat_result.st_ctime = 1000
                                    mock_stat_result.st_mtime = 2000
                                    mock_stat_result.st_size = 100
                                    mock_stat.return_value = mock_stat_result
                                    
                                    result = await tool.execute({
                                        "directory_path": "/path/to/dir",
                                        "recursive": True,
                                        "max_depth": 2  # Only go two levels deep
                                    })
                                    
                                    assert result["success"] is True
                                    assert result["recursive"] is True
                                    
                                    # Count items
                                    file_items = [item for item in result["items"] if item["type"] == "file"]
                                    dir_items = [item for item in result["items"] if item["type"] == "directory"]
                                    
                                    # Should include files from root and first level, but not from subdir3
                                    assert len(file_items) == 4  # file1.txt, file2.txt, file3.txt, file4.txt
                                    
                                    # Should include directories from root and first level
                                    assert len(dir_items) == 3  # subdir1, subdir2, subdir3
