/**
 * v0 by Vercel.
 * @see https://v0.dev/t/yXUK5JvR2xN
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
export default function Component() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-[#F2F2F2] to-[#E6E6E6]">
      <header className="bg-[#1C3144] py-4 px-6 md:px-12 lg:px-20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <img src="/placeholder.svg" alt="BIAS Logo" className="h-8 w-8" />
            <h1 className="text-2xl font-bold text-white">BIAS</h1>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <a href="#" className="text-white hover:text-[#FFA500] transition-colors">
              About
            </a>
            <a href="#" className="text-white hover:text-[#FFA500] transition-colors">
              Team
            </a>
            <a href="#" className="text-white hover:text-[#FFA500] transition-colors">
              Contact
            </a>
          </nav>
          <button className="md:hidden text-white hover:text-[#FFA500] transition-colors">
            <MenuIcon className="h-6 w-6" />
          </button>
        </div>
      </header>
      <main className="flex-1">
        <section id="hero" className="bg-[#1C3144] py-12 px-6 md:px-12 lg:px-20">
          <div className="max-w-4xl mx-auto flex flex-col items-center text-center gap-6">
            <h2 className="text-3xl font-bold text-white">Welcome to BIAS</h2>
            <p className="text-lg text-white">
              BIAS is a revolutionary wheelchair controlled by brain signals, designed to improve the quality of life
              for individuals with mobility challenges.
            </p>
            <a
              href="#"
              className="inline-flex items-center justify-center bg-[#FFA500] text-white font-medium py-3 px-6 rounded-md hover:bg-[#E69500] transition-colors"
            >
              Learn More
            </a>
          </div>
        </section>
        <section id="about" className="py-12 px-6 md:px-12 lg:px-20">
          <div className="max-w-4xl mx-auto flex flex-col items-center gap-6">
            <h2 className="text-3xl font-bold text-[#1C3144]">How BIAS Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-2xl font-bold text-[#1C3144] mb-2">Raspberry Pi 4</h3>
                <p className="text-[#4A4A4A]">
                  The Raspberry Pi 4 is the brain of the BIAS system, processing the brain signals and controlling the
                  wheelchair's movements.
                </p>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-[#1C3144] mb-2">Brain Electrodes</h3>
                <p className="text-[#4A4A4A]">
                  The electrodes placed on the user's head capture the brain signals, which are then transmitted to the
                  Raspberry Pi for processing.
                </p>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-[#1C3144] mb-2">Motor System</h3>
                <p className="text-[#4A4A4A]">
                  The wheelchair's motor system is controlled by the Raspberry Pi, allowing the user to navigate the
                  wheelchair using their brain signals.
                </p>
              </div>
            </div>
          </div>
        </section>
        <section id="team" className="bg-[#1C3144] py-12 px-6 md:px-12 lg:px-20">
          <div className="max-w-4xl mx-auto flex flex-col items-center gap-6">
            <h2 className="text-3xl font-bold text-white">Meet the Team</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-md shadow-md p-4 flex flex-col items-center gap-2">
                <img src="/placeholder.svg" alt="Santiago Sojka" className="w-24 h-24 rounded-full object-cover" />
                <h3 className="text-lg font-bold text-[#1C3144]">Santiago Sojka</h3>
                <p className="text-[#4A4A4A]">Project Manager</p>
              </div>
              <div className="bg-white rounded-md shadow-md p-4 flex flex-col items-center gap-2">
                <img src="/placeholder.svg" alt="Danilo Diaz" className="w-24 h-24 rounded-full object-cover" />
                <h3 className="text-lg font-bold text-[#1C3144]">Danilo Diaz</h3>
                <p className="text-[#4A4A4A]">Hardware Engineer</p>
              </div>
              <div className="bg-white rounded-md shadow-md p-4 flex flex-col items-center gap-2">
                <img src="/placeholder.svg" alt="Nicolas Adell" className="w-24 h-24 rounded-full object-cover" />
                <h3 className="text-lg font-bold text-[#1C3144]">Nicolas Adell</h3>
                <p className="text-[#4A4A4A]">Software Engineer</p>
              </div>
              <div className="bg-white rounded-md shadow-md p-4 flex flex-col items-center gap-2">
                <img src="/placeholder.svg" alt="Luciano Montenegro" className="w-24 h-24 rounded-full object-cover" />
                <h3 className="text-lg font-bold text-[#1C3144]">Luciano Montenegro</h3>
                <p className="text-[#4A4A4A]">Electrical Engineer</p>
              </div>
              <div className="bg-white rounded-md shadow-md p-4 flex flex-col items-center gap-2">
                <img src="/placeholder.svg" alt="Gil Soria Ian" className="w-24 h-24 rounded-full object-cover" />
                <h3 className="text-lg font-bold text-[#1C3144]">Gil Soria Ian</h3>
                <p className="text-[#4A4A4A]">Mechanical Engineer</p>
              </div>
            </div>
          </div>
        </section>
        <section id="contact" className="py-12 px-6 md:px-12 lg:px-20">
          <div className="max-w-4xl mx-auto flex flex-col items-center gap-6">
            <h2 className="text-3xl font-bold text-[#1C3144]">Contact Us</h2>
            <div className="flex flex-col items-center gap-4">
              <p className="text-[#4A4A4A]">
                Email:{" "}
                <a href="#" className="text-[#FFA500] hover:underline">
                  bias.project@gmail.com
                </a>
              </p>
              <p className="text-[#4A4A4A]">
                Instagram:{" "}
                <a href="#" target="_blank" rel="noopener noreferrer" className="text-[#FFA500] hover:underline">
                  @proyecto.bias
                </a>
              </p>
            </div>
          </div>
        </section>
      </main>
      <footer className="bg-[#1C3144] py-4 px-6 md:px-12 lg:px-20">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <p className="text-white text-sm">&copy; 2023 BIAS. All rights reserved.</p>
          <div className="flex items-center gap-4">
            <a href="#" className="text-white hover:text-[#FFA500] transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-white hover:text-[#FFA500] transition-colors">
              Terms of Service
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}

function MenuIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="4" x2="20" y1="12" y2="12" />
      <line x1="4" x2="20" y1="6" y2="6" />
      <line x1="4" x2="20" y1="18" y2="18" />
    </svg>
  )
}
