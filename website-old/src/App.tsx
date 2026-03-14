import React from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import { LocaleProvider } from "./i18n";
import CourseRoadmap from "./components/CourseRoadmap";
import LessonPage from "./pages/LessonPage";

const App: React.FC = () => {
  return (
    <LocaleProvider>
      <HashRouter>
        <Routes>
          <Route path="/" element={<CourseRoadmap />} />
          <Route path="/phase/:phaseId/lesson/:lessonId" element={<LessonPage />} />
        </Routes>
      </HashRouter>
    </LocaleProvider>
  );
};

export default App;
