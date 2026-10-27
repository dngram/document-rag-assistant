pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        PROJECT_DIR = 'C:\\Projects\\document-rag-assistant'
        PYTHON = 'C:\\Projects\\document-rag-assistant\\.venv\\Scripts\\python.exe'
    }

    stages {
        stage('Verify Environment') {
            steps {
                dir("${PROJECT_DIR}") {
                    bat '"%PYTHON%" --version'
                    bat '"%PYTHON%" -m pip --version'
                    bat 'git status --short'
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                dir("${PROJECT_DIR}") {
                    bat '"%PYTHON%" -m pip install -r requirements.txt'
                }
            }
        }

        stage('Run Tests') {
            steps {
                dir("${PROJECT_DIR}") {
                    bat '"%PYTHON%" -m pytest -v'
                }
            }
        }
    }

    post {
        success {
            echo 'CI pipeline passed: all tests succeeded.'
        }

        failure {
            echo 'CI pipeline failed. Review the console output.'
        }

        always {
            echo 'CI pipeline execution completed.'
        }
    }
}