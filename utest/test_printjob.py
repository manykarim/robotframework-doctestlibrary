"""Unit tests for PrintJob module."""

from DocTest.PrintJob import PrintJob


class TestPrintJob:
    """Test cases for the PrintJob class."""

    def test_printjob_initialization(self):
        """Test PrintJob object initialization with basic parameters."""
        properties = [
            {"property": "paper_source", "value": "1"},
            {"property": "copies", "value": "2"},
        ]
        job = PrintJob("pcl", properties)

        assert job.jobtype == "pcl"
        assert job.properties == properties
        assert len(job.properties) == 2

    def test_printjob_initialization_empty_properties(self):
        """Test PrintJob object initialization with empty properties."""
        job = PrintJob("postscript", [])

        assert job.jobtype == "postscript"
        assert job.properties == []

    def test_printjob_initialization_none_properties(self):
        """Test PrintJob object initialization with None properties."""
        job = PrintJob("pcl", None)

        assert job.jobtype == "pcl"
        assert job.properties is None

    def test_printjob_initialization_complex_properties(self):
        """Test PrintJob object initialization with complex properties."""
        properties = [
            {
                "page": "1",
                "property": "pcl_commands",
                "value": [
                    {"page": "1", "property": "paper_source", "value": "2"},
                    {"page": "1", "property": "copies", "value": "1"},
                    {"page": "1", "property": "duplex", "value": "0"},
                ],
            }
        ]
        job = PrintJob("pcl", properties)

        assert job.jobtype == "pcl"
        assert job.properties == properties
        assert job.properties[0]["property"] == "pcl_commands"
        assert len(job.properties[0]["value"]) == 3

    def test_printjob_different_jobtypes(self):
        """Test PrintJob with different job types."""
        pcl_job = PrintJob("pcl", [])
        ps_job = PrintJob("postscript", [])
        pdf_job = PrintJob("pdf", [])

        assert pcl_job.jobtype == "pcl"
        assert ps_job.jobtype == "postscript"
        assert pdf_job.jobtype == "pdf"

    def test_printjob_properties_modification(self):
        """Test that PrintJob properties can be modified after initialization."""
        properties = [{"property": "test", "value": "initial"}]
        job = PrintJob("test", properties)

        # Modify the properties
        job.properties.append({"property": "new", "value": "added"})

        assert len(job.properties) == 2
        assert job.properties[1]["property"] == "new"
        assert job.properties[1]["value"] == "added"
