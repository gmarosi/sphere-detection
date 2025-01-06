#pragma once

#include <string>
#include <vector>
#include <windows.h>

typedef std::pair<std::wstring, std::vector<std::pair<std::wstring, std::wstring>>> MemoryNames;

//class responsible for managing shared memory on Windows OS
class SHMManager
{
	typedef std::pair<std::wstring, std::wstring> SHMNamePair;
	typedef std::pair<HANDLE, HANDLE> SHMHandlePair;

public:
	/**
	 * \brief Creates a named shared memory managing object.
	 * Has 1 sync, N double buffers
	 * \param syncName name of memory containing buffer flag
	 * \param dataNames name of the buffers in pairs
	 * \param syncSize size of memory containing buffer flag
	 * \param dataSize size of memory containing point data
	 */
	SHMManager(const std::wstring& syncName, const std::vector<SHMNamePair>& dataNames, const size_t syncSize, const size_t dataSize);
	virtual ~SHMManager();

	/**
	 * \brief Initializes map file
	 * \param hMapFile address of handle
	 * \param memName name of memory
	 * \param memSize size of memory
	 */
	static void initMapFile(HANDLE& hMapFile, const std::wstring& memName, const size_t memSize);

	/**
	 * \brief Reads sync flag's value
	 * \return flag
	 */
	int			readSync();

	/**
	 * \brief Reads data from correct buffer
	 * \param dest destination address
	 * \param pairIdx index of buffer pair
	 */
	void		readData(void* dest, const int pairIdx = 0);

	/**
	 * \brief Checks if flag has changed from previous state
	 * \return flag changed
	 */
	bool		hasBufferChanged();

	/**
	 * \brief Gets how many buffer pairs are in the manager
	 * \return number of memory pairs
	 */
	int			bufferPairCount() const;

protected:
	int							m_currSyncFlag;

	HANDLE						m_hSyncMapFile;		//sync
	std::vector<SHMHandlePair>	m_hDataMapFiles;	//stores data

	size_t						m_syncMemSize;
	size_t						m_dataMemSize;

	std::wstring				m_syncMemName;
	std::vector<SHMNamePair>	m_dataMemNames;
};