import { SignIn } from '@clerk/nextjs';
import { SignOutButton } from '@clerk/nextjs';

export default function Page() {
  return (
    <div className='flex justify-center items-center h-screen'>
      <SignIn />
    </div>
  );
}
