@echo off
chcp 65001 >nul
echo Checking database tables...
echo.

sqlite3.exe build\bin\Release\demo_basic.db ".tables"

echo.
echo Table schema:
sqlite3.exe build\bin\Release\demo_basic.db ".schema"

echo.
echo Sample data:
sqlite3.exe build\bin\Release\demo_basic.db "SELECT rowid, content, length(embedding) as vec_bytes FROM messages LIMIT 3;"

pause
