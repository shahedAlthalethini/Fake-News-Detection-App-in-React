import React from "react";

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-green-100 via-white to-blue-100 text-gray-800 font-sans">
      {/* Header */}
      <header className="bg-green-600 text-white text-center py-6 shadow-md">
        <h1 className="text-4xl font-bold tracking-wide">فلسطين عين الحقيقة</h1>
        <p className="mt-2 text-lg">منصة لرصد الأخبار الفلسطينية بصدق ووضوح</p>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto p-6">
        <div className="max-w-3xl mx-auto bg-white p-6 rounded-2xl shadow-lg">
          <h2 className="text-2xl font-semibold text-center mb-6 text-green-700">أدخل الخبر هنا</h2>
          <form className="space-y-4">
            <div>
              <label className="block mb-1 font-medium">عنوان الخبر</label>
              <input
                type="text"
                className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                placeholder="أدخل عنوان الخبر"
              />
            </div>
            <div>
              <label className="block mb-1 font-medium">نص الخبر</label>
              <textarea
                rows="6"
                className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500"
                placeholder="اكتب تفاصيل الخبر هنا"
              ></textarea>
            </div>
            <button
              type="submit"
              className="w-full bg-green-600 text-white py-3 rounded-xl hover:bg-green-700 transition duration-300 font-semibold"
            >
              إرسال الخبر
            </button>
          </form>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-green-600 text-white text-center py-4">
        <p>© 2025 فلسطين عين الحقيقة - جميع الحقوق محفوظة</p>
      </footer>
    </div>
  );
}

export default App;
