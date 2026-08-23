import http from 'k6/http';

export const options = {
    vus: 50,
    duration: '500s',
}

export default () => {
    // http.get('http://54.145.145.235:8501');
    http.get('http://172.31.46.143:8501');
};

// comment