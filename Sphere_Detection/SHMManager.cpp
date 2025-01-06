#include "SHMManager.h"

#include <iostream>

SHMManager::SHMManager(const std::wstring& syncName, const std::vector<SHMNamePair>& dataNames, const size_t syncSize, const size_t dataSize)
{
	//init names
	m_syncMemName = syncName;

	for (const SHMNamePair& namePair : dataNames)
	{
		m_dataMemNames.push_back(namePair);
	}

	//init sizes
	m_syncMemSize = syncSize;
	m_dataMemSize = dataSize;


	//init memory
	initMapFile(m_hSyncMapFile, m_syncMemName, m_syncMemSize);
	m_hDataMapFiles.resize(m_dataMemNames.size());
	for (size_t i = 0; i < m_dataMemNames.size(); ++i)
	{
		initMapFile(m_hDataMapFiles[i].first, m_dataMemNames[i].first, m_dataMemSize);	//buffer/1
		initMapFile(m_hDataMapFiles[i].second, m_dataMemNames[i].second, m_dataMemSize);	//buffer/2
	}

	m_currSyncFlag = 0; //start with buffer/1
}

SHMManager::~SHMManager()
{
	CloseHandle(m_hSyncMapFile);
	for (const SHMHandlePair& handlePair : m_hDataMapFiles)
	{
		CloseHandle(handlePair.first);
		CloseHandle(handlePair.second);
	}
}

void SHMManager::initMapFile(HANDLE& hMapFile, const std::wstring& memName, const size_t memSize)
{
	hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, memName.c_str());

	//create file mapping if could not open (LiDAR_to_SHM not running for example)
	if (!hMapFile)
	{
		hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, memSize, memName.c_str());
	}
}

int SHMManager::readSync()
{
	const LPVOID pBuf = MapViewOfFile(m_hSyncMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int));

	if (!pBuf)
	{
		std::cerr << "SHMManager: buffer is NULL after reading buffer flag from shared memory" << std::endl;
		return -1;
	}

	int flag;
	memcpy(&flag, pBuf, m_syncMemSize);
	UnmapViewOfFile(pBuf);

	return flag;
}

void SHMManager::readData(void* dest, const int pairIdx)
{
	HANDLE dataMapFile = nullptr;
	switch (m_currSyncFlag)
	{
	case 0:
		dataMapFile = m_hDataMapFiles[pairIdx].first;
		break;
	case 1:
		dataMapFile = m_hDataMapFiles[pairIdx].second;
		break;
	default:
		std::cerr << "Invalid sync flag value!" << std::endl;
		return;
	}
	const LPVOID pBuf = MapViewOfFile(dataMapFile, FILE_MAP_ALL_ACCESS, 0, 0, m_dataMemSize);

	if (!pBuf)
	{
		std::cerr << "SHMManager: buffer is NULL after reading point cloud data from shared memory" << std::endl;
		return;
	}

	memcpy(dest, pBuf, m_dataMemSize);
	UnmapViewOfFile(pBuf);
}

bool SHMManager::hasBufferChanged()
{
	const int readFlag = readSync();

	const bool isFlagChanged = (m_currSyncFlag != readFlag);

	//update flag if changed
	if (isFlagChanged)
		m_currSyncFlag = readFlag;

	return isFlagChanged;
}

int SHMManager::bufferPairCount() const
{
	return m_dataMemNames.size();
}
