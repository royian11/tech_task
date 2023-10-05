import { useState, useEffect } from 'react';
import Papa from 'papaparse';

function App() {
  const [data, setData] = useState([]);
  const [selectedTitle, setSelectedTitle] = useState('');
  const [tableData, setTableData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('/clean_data.csv');
      const reader = response.body.getReader();
      const result = await reader.read();
      const decoder = new TextDecoder('utf-8');
      const csv = decoder.decode(result.value);
      const parsedData = Papa.parse(csv, { header: true });
      setData(parsedData.data);
    }

    fetchData();
  }, []);

  const handleTitleChange = (e) => {
    const selectedTitle = e.target.value;
    setSelectedTitle(selectedTitle);

    // Filter data based on selected title
    const filteredData = data.filter((row) => row.document_title === selectedTitle);
    setTableData(filteredData);
  };

  return (
    <div>
      <div>
        <h1>CSV Reader</h1>
        <select value={selectedTitle} onChange={handleTitleChange} >
          <option value="">Select Document Title</option>
          {Array.from(new Set(data.map((row) => row.document_title))).map((title) => (
            <option key={title} value={title}>
              {title}
            </option>
          ))}
        </select>

        {selectedTitle && (
          <table>
            <thead>
              <tr>
                <th>question_text</th>
                <th>long_answer_text</th>
                <th>yes_no_answer</th>
                <th>short_answer1</th>
                <th>short_answer3</th>
                {/* Add more headers as needed */}
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, index) => (
                <tr key={index}>
                  <td>{row.question_text}</td>
                  <td>{row.long_answer_text}</td>
                  <td>{row.yes_no_answer}</td>
                  <td>{row.short_answer1}</td>
                  <td>{row.short_answer3}</td>
                  {/* Add more columns as needed */}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

    </div>
  );
}

export default App;
